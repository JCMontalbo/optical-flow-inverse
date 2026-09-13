"""Train one cell on MSD Heart and score 3D Dice on the val volumes.

    python experiments/heart/train.py --budget 2 --arm flow --seed 0

Arms: none | plausible | elastic | flow | propagate   (docs/downstream_heart_plan.md)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.benchmark = True

import importlib.util

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from data import BUDGETS, H, W, labelled_indices, load_volume, split  # noqa: E402
from flowaug import elastic, flow_family, intensity_jitter, plausible_affine  # noqa: E402

_spec = importlib.util.spec_from_file_location("unet", HERE.parent / "downstream" / "unet.py")
_unet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_unet)
UNet, bce_dice = _unet.UNet, _unet.bce_dice

from ofi.labels import propagate_label  # noqa: E402

FLOWS = HERE / "flows"
RESULTS = HERE / "results.csv"
ARMS = ["none", "plausible", "elastic", "flow", "propagate"]
OFFSETS = (1, -1, 2, -2)


def build_sets(budget: int, arm: str):
    train, _ = split()
    real_x, real_y, centers, flows, es_x, es_y = [], [], [], [], [], []
    nb_x, nb_y = [], []
    n_flows_ok = 0
    for v in train:
        img, m, _ = load_volume(v)
        sel = labelled_indices(m, budget)
        have = [k for k in range(1, len(m) - 1) if m[k].any() and k not in sel]
        if not have:  # budget "all": reserve 5 labelled slices per volume for early stopping
            es = np.random.default_rng(7 + sum(map(ord, v))).choice(sel, size=5, replace=False)
            sel = [k for k in sel if k not in set(es.tolist())]
        else:
            es = np.random.default_rng(7 + sum(map(ord, v))).choice(have, size=min(5, len(have)), replace=False)
        es_x += [img[k] for k in es]
        es_y += [m[k] for k in es]
        need_flows = arm in ("flow", "propagate")
        z = np.load(FLOWS / f"{v}.npz") if need_flows else None
        for k in sel:
            real_x.append(img[k])
            real_y.append(m[k])
            ys, xs = np.where(m[k])
            centers.append((xs.mean(), ys.mean()))
            fl = np.zeros((4, 2, H, W), np.float16)
            for i, d in enumerate(OFFSETS):
                key = f"{k}_{k + d}"
                if need_flows and key in z.files:
                    fl[i] = z[key]
                    n_flows_ok += 1
                    if arm == "propagate":
                        u, vv = z[key].astype(np.float32)
                        nb_x.append(img[k + d])
                        nb_y.append(propagate_label(m[k], u, vv))
            flows.append(fl)
    return dict(
        real_x=np.stack(real_x), real_y=np.stack(real_y), centers=np.array(centers, np.float32), flows=np.stack(flows),
        es_x=np.stack(es_x), es_y=np.stack(es_y),
        nb_x=np.stack(nb_x) if nb_x else np.zeros((0, H, W), np.float32), nb_y=np.stack(nb_y) if nb_y else np.zeros((0, H, W), bool),
        n_flows_ok=n_flows_ok,
    )


def tens(a, device, dtype=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)


@torch.no_grad()
def dice3d(model, device, bs=32):
    model.eval()
    _, val = split()
    per = {}
    for v in val:
        img, m, _ = load_volume(v)
        preds = []
        for i in range(0, len(img), bs):
            x = tens(img[i : i + bs], device).unsqueeze(1)
            preds.append((model(x) > 0).squeeze(1).cpu())
        p = torch.cat(preds).numpy()
        inter = (p & m).sum()
        per[v] = float(2 * inter / max(p.sum() + m.sum(), 1))
    model.train()
    return float(np.mean(list(per.values()))), per


@torch.no_grad()
def es_dice(model, es_x, es_y, bs=32):
    model.eval()
    num, den = 0.0, 0.0
    for i in range(0, len(es_x), bs):
        p = model(es_x[i : i + bs]) > 0
        t = es_y[i : i + bs] > 0.5
        num += 2 * (p & t).sum().item()
        den += (p.sum() + t.sum()).item()
    model.train()
    return num / max(den, 1)


def run(budget: int, arm: str, seed: int, steps: int = 3000, batch: int = 16, device="cuda", keep_model: bool = False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    S = build_sets(budget, arm)
    rx = tens(S["real_x"], device).unsqueeze(1)
    ry = tens(S["real_y"], device).unsqueeze(1)
    cen = tens(S["centers"], device)
    fl = tens(S["flows"], device, torch.float16)  # [N,4,2,H,W]
    valid = fl.flatten(2).abs().amax(-1) > 0  # [N,4]
    es_x = tens(S["es_x"], device).unsqueeze(1)
    es_y = tens(S["es_y"], device).unsqueeze(1)
    if arm == "propagate" and len(S["nb_x"]):
        gx = tens(S["nb_x"], device).unsqueeze(1)
        gy = tens(S["nb_y"], device).unsqueeze(1)
    else:
        gx = gy = None
    n_real = len(rx)

    model = UNet(cin=1).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best, best_state, best_step = -1.0, None, 0
    disp = []
    t0 = time.time()
    for step in range(1, steps + 1):
        ir = torch.randint(0, n_real, (batch,), generator=gen)
        x, y = rx[ir], ry[ir]
        if gx is not None:  # propagate: half the batch from the neighbour pairs
            take = (torch.rand(batch, generator=gen) < 0.5).to(device).view(-1, 1, 1, 1)
            ig = torch.randint(0, len(gx), (batch,), generator=gen)
            x = torch.where(take, gx[ig], x)
            y = torch.where(take, gy[ig], y)
        if arm == "plausible":
            x, y = plausible_affine(x, y, gen)
        elif arm == "elastic":
            x, y = elastic(x, y, gen)
        elif arm == "flow":
            # pick one accepted neighbour flow per sample (zero flow if none accepted)
            r = torch.rand(batch, 4, generator=gen).to(device) * valid[ir]
            slot = r.argmax(1)
            f = fl[ir, slot].float() * valid[ir].any(1).view(-1, 1, 1, 1)
            x, y = flow_family(x, y, f, cen[ir], gen)
        if arm != "none":
            x = intensity_jitter(x, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = bce_dice(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 150 == 0 or step == steps:
            d = es_dice(model, es_x, es_y)
            if d > best:
                best, best_step = d, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    dice, per = dice3d(model, device)
    row = dict(budget=budget, arm=arm, seed=seed, dice=round(dice, 4), es_dice=round(best, 4), best_step=best_step, n_real=n_real,
               n_flows_ok=S["n_flows_ok"], n_nb=0 if gx is None else len(gx), minutes=round((time.time() - t0) / 60, 2),
               per_volume=json.dumps({k: round(v, 4) for k, v in per.items()}))
    if keep_model:
        row["model"] = model
    return row


def append_to(path, row):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, choices=BUDGETS, required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    row = run(a.budget, a.arm, a.seed, steps=a.steps)
    print({k: v for k, v in row.items() if k != "per_volume"})
    if not a.no_save:
        append_to(RESULTS, row)
