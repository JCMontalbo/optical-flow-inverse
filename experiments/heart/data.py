"""MSD Task02 Heart: volumes -> cropped, normalised slice stacks, cached; split; label budgets."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("MSD_HEART", r"C:\Users\Socce\data\Task02_Heart"))
CACHE = Path(__file__).resolve().parent / "cache"
CROP = (slice(48, 240), slice(72, 264))  # 192 x 192, covers the atrium in every volume
H = W = 192
MARGIN = 3
BUDGETS = [1, 2, 4, 999]  # 999 = all labelled slices


def volume_ids() -> list[str]:
    return sorted(p.name[:-7] for p in (ROOT / "imagesTr").glob("la_*.nii.gz"))


def split():
    ids = volume_ids()
    val = ids[2::3]  # every third in sorted order
    train = [v for v in ids if v not in val]
    return train, val


def load_volume(vid: str):
    """(slices float32 [n, H, W] in [0,1], masks bool [n, H, W], z0) restricted to the labelled range +- MARGIN."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{vid}.npz"
    if f.exists():
        with np.load(f) as z:
            return z["img"], z["mask"], int(z["z0"])
    import nibabel as nib

    img = nib.load(ROOT / "imagesTr" / f"{vid}.nii.gz").get_fdata().astype(np.float32)
    lab = nib.load(ROOT / "labelsTr" / f"{vid}.nii.gz").get_fdata() > 0
    zs = np.where(lab.any(axis=(0, 1)))[0]
    z0, z1 = max(0, zs.min() - MARGIN), min(img.shape[2] - 1, zs.max() + MARGIN)
    img = img[CROP[0], CROP[1], z0 : z1 + 1]
    lab = lab[CROP[0], CROP[1], z0 : z1 + 1]
    hi = np.percentile(img, 99.5)
    img = np.clip(img / max(hi, 1e-6), 0, 1).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))  # [n, H, W]
    lab = np.transpose(lab, (2, 0, 1))
    np.savez_compressed(f, img=img, mask=lab, z0=z0)
    return img, lab, int(z0)


def labelled_indices(masks: np.ndarray, per_volume: int) -> list[int]:
    """Evenly spaced slices among those that contain the atrium (never the first/last slice of the pool)."""
    have = [k for k in range(1, len(masks) - 1) if masks[k].any()]
    if per_volume >= len(have):
        return have
    pos = np.linspace(0, len(have) - 1, per_volume + 2)[1:-1] if per_volume > 1 else np.array([(len(have) - 1) / 2])
    return sorted({have[int(round(p))] for p in pos})


def all_labelled(train: list[str]) -> dict[str, set[int]]:
    out = {}
    for v in train:
        _, m, _ = load_volume(v)
        out[v] = set()
        for b in BUDGETS:
            out[v].update(labelled_indices(m, b))
    return out


if __name__ == "__main__":
    tr, va = split()
    print("train", tr)
    print("val  ", va)
    n = 0
    for v in tr + va:
        img, m, z0 = load_volume(v)
        n += len(img)
        print(f"  {v}: {len(img)} slices from z={z0}, atrium on {m.any(axis=(1, 2)).sum()}, mean intensity {img.mean():.3f}")
    print("total slices", n)
