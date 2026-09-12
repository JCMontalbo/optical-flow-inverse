"""Train one cell of the grid and score it on the DAVIS 2016 val videos.

    python experiments/downstream/train.py --budget 2 --arm flow --seed 0

Arms: none | standard | flow | standard+flow | propagate   (see docs/downstream_plan.md)

Same recipe for every arm: small U-Net, BCE + Dice, Adam 1e-3 with cosine decay,
batch 8, 4,000 steps for every cell, early stopping on the *training*
videos' non-selected frames (never on val). Real and generated samples are
drawn with equal probability per batch. Appends one row to results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import BUDGETS, H, W, labelled_indices, load_video, videos  # noqa: E402
from unet import UNet, bce_dice  # noqa: E402

HERE = Path(__file__).resolve().parent
GEN = HERE / "generated"
RESULTS = HERE / "results.csv"
ARMS = ["none", "standard", "flow", "standard+flow", "propagate"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- data


def build_sets(budget: int, arm: str, seed: int):
    """Returns (real_imgs, real_masks, gen_imgs, gen_masks, es_imgs, es_masks) as uint8/bool arrays."""
    train = videos("train")
    rng = np.random.default_rng(seed)
    real_i, real_m, gen_i, gen_m, es_i, es_m = [], [], [], [], [], []
    for v in train:
        frames, masks = load_video(v)
        sel = labelled_indices(len(frames), budget)
        real_i += [frames[k] for k in sel]
        real_m += [masks[k] for k in sel]
        # early-stopping frames: 3 per video, drawn from the non-selected frames, same for every arm
        pool = [k for k in range(len(frames)) if k not in sel]
        es = np.random.default_rng(1000 + hash_str(v)).choice(pool, size=min(3, len(pool)), replace=False)
        es_i += [frames[k] for k in es]
        es_m += [masks[k] for k in es]
        if arm in ("flow", "standard+flow", "propagate"):
            for k in sel:
                with np.load(GEN / f"{v}_{k:05d}.npz") as z:
                    if arm == "propagate":
                        gen_i += list(z["nb_imgs"])
                        gen_m += list(z["nb_masks"])
                    else:
                        gen_i += list(z["imgs"])
                        gen_m += list(z["masks"])
    stack = lambda xs, dt: np.stack(xs).astype(dt) if xs else np.zeros((0, H, W) + ((3,) if dt == np.uint8 else ()), dt)  # noqa: E731
    return stack(real_i, np.uint8), stack(real_m, bool), stack(gen_i, np.uint8), stack(gen_m, bool), stack(es_i, np.uint8), stack(es_m, bool)


def hash_str(s: str) -> int:
    import zlib

    return zlib.crc32(s.encode())


def to_tensor(imgs_u8, masks_b, device):
    """Kept as uint8 / bool on the GPU; converted per batch by ``as_float``."""
    x = torch.from_numpy(np.ascontiguousarray(imgs_u8)).permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(np.ascontiguousarray(masks_b)).unsqueeze(1)
    return x.to(device), y.to(device)


def as_float(x_u8, y_b):
    return x_u8.float().div_(255.0), y_b.float()


def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def standard_augment(x, y, gen):
    """Random flip, rotation +-15 deg, scale 0.8-1.2, shear +-10 deg (image bilinear, mask nearest), colour jitter."""
    n = x.shape[0]
    dev = x.device
    flip = (torch.rand(n, generator=gen, device="cpu") < 0.5).to(dev)
    ang = (torch.rand(n, generator=gen) * 30 - 15).to(dev) * math.pi / 180
    sc = (torch.rand(n, generator=gen) * 0.4 + 0.8).to(dev)
    sh = (torch.rand(n, generator=gen) * 20 - 10).to(dev) * math.pi / 180
    cos, sin = torch.cos(ang), torch.sin(ang)
    # affine matrix in normalised coords; account for aspect so rotation is not distorted
    asp = W / H
    a = torch.zeros(n, 2, 3, device=dev)
    a[:, 0, 0] = cos / sc
    a[:, 0, 1] = (-sin + torch.tan(sh)) / sc / asp
    a[:, 1, 0] = sin / sc * asp
    a[:, 1, 1] = cos / sc
    a[:, 0, 0] = torch.where(flip, -a[:, 0, 0], a[:, 0, 0])
    a[:, 1, 0] = torch.where(flip, -a[:, 1, 0], a[:, 1, 0])
    grid = F.affine_grid(a, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    y = F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    b = (torch.rand(n, 1, 1, 1, generator=gen) * 0.4 - 0.2).to(dev)
    c = (torch.rand(n, 1, 1, 1, generator=gen) * 0.4 + 0.8).to(dev)
    x = ((x - 0.5) * c + 0.5 + b).clamp_(0, 1)
    return x, y


# ----------------------------------------------------------------------------- eval


@torch.no_grad()
def iou_batch(logits, y):
    p = logits > 0
    t = y > 0.5
    inter = (p & t).sum((1, 2, 3)).float()
    union = (p | t).sum((1, 2, 3)).float()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(inter))


@torch.no_grad()
def evaluate_val(model, device, bs=16):
    """DAVIS J-mean: per-frame IoU -> mean per video -> mean over videos."""
    model.eval()
    per_video = {}
    for v in videos("val"):
        frames, masks = load_video(v)
        ious = []
        for i in range(0, len(frames), bs):
            x, y = as_float(*to_tensor(frames[i : i + bs], masks[i : i + bs], device))
            ious.append(iou_batch(model(normalize(x)), y).cpu())
        per_video[v] = float(torch.cat(ious).mean())
    model.train()
    return float(np.mean(list(per_video.values()))), per_video


@torch.no_grad()
def evaluate_es(model, es_x, es_y, bs=16):
    model.eval()
    ious = [iou_batch(model(normalize(as_float(es_x[i : i + bs], es_y[i : i + bs])[0])), es_y[i : i + bs].float()) for i in range(0, len(es_x), bs)]
    model.train()
    return float(torch.cat(ious).mean())


# ----------------------------------------------------------------------------- train


def run(budget: int, arm: str, seed: int, max_steps: int = 4000, batch: int = 8, device="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    real_i, real_m, gen_i, gen_m, es_i, es_m = build_sets(budget, arm, seed)
    rx, ry = to_tensor(real_i, real_m, device)
    gx, gy = to_tensor(gen_i, gen_m, device) if len(gen_i) else (None, None)
    es_x, es_y = to_tensor(es_i, es_m, device)
    use_std = arm in ("standard", "standard+flow")
    n_real, n_gen = len(rx), 0 if gx is None else len(gx)

    model = UNet().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    total = max_steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    best_es, best_state, best_step = -1.0, None, 0
    t0 = time.time()
    for step in range(1, total + 1):
        # equal probability real / generated per sample
        if gx is not None:
            take_gen = torch.rand(batch, generator=gen) < 0.5
            ir = torch.randint(0, n_real, (batch,), generator=gen)
            ig = torch.randint(0, n_gen, (batch,), generator=gen)
            x = torch.where(take_gen.view(-1, 1, 1, 1).to(device), gx[ig], rx[ir])
            y = torch.where(take_gen.view(-1, 1, 1, 1).to(device), gy[ig], ry[ir])
        else:
            ir = torch.randint(0, n_real, (batch,), generator=gen)
            x, y = rx[ir], ry[ir]
        x, y = as_float(x, y)
        if use_std:
            x, y = standard_augment(x, y, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(normalize(x))
        loss = bce_dice(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0 or step == total:
            es = evaluate_es(model, es_x, es_y)
            if es > best_es:
                best_es, best_step = es, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    jmean, per_video = evaluate_val(model, device)
    return dict(
        budget=budget, arm=arm, seed=seed, jmean=round(jmean, 4), es_iou=round(best_es, 4), best_step=best_step,
        steps=total, n_real=n_real, n_gen=n_gen, minutes=round((time.time() - t0) / 60, 2), per_video=json.dumps({k: round(v, 4) for k, v in per_video.items()}),
    )


def append_result(row: dict):
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, choices=BUDGETS, required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    row = run(args.budget, args.arm, args.seed, max_steps=args.max_steps)
    print({k: v for k, v in row.items() if k != "per_video"})
    if not args.no_save:
        append_result(row)


if __name__ == "__main__":
    main()
