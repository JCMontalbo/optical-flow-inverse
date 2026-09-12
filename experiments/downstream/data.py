"""DAVIS 2016 loading, resizing, caching, split and label-budget selection.

Frames and masks are resized once to H x W (default 256 x 448) and cached as
uint8 / bool arrays per video in ``cache/``, so every later step is fast.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

H, W = 256, 448
DAVIS_ROOT = Path(os.environ.get("DAVIS_ROOT", r"C:\Users\Socce\data\DAVIS"))
CACHE = Path(__file__).resolve().parent / "cache"


def video_list(split: str) -> list[str]:
    return [ln.strip() for ln in (DAVIS_ROOT / "ImageSets" / "480p" / f"{split}.txt").read_text().splitlines() if ln.strip()]


def _names_from_imageset(split: str) -> list[str]:
    # DAVIS 2016 ImageSets lines look like "/JPEGImages/480p/bear/00000.jpg /Annotations/480p/bear/00000.png"
    names = []
    for ln in video_list(split):
        v = ln.split()[0].split("/")[3]
        if v not in names:
            names.append(v)
    return names


def videos(split: str) -> list[str]:
    return _names_from_imageset(split)


def load_video(name: str):
    """(frames uint8 [n, H, W, 3], masks bool [n, H, W]) for one video, cached."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{name}.npz"
    if f.exists():
        with np.load(f) as z:
            return z["frames"], z["masks"]
    jdir = DAVIS_ROOT / "JPEGImages" / "480p" / name
    adir = DAVIS_ROOT / "Annotations" / "480p" / name
    frames, masks = [], []
    for jp in sorted(jdir.glob("*.jpg")):
        img = Image.open(jp).convert("RGB").resize((W, H), Image.BILINEAR)
        m = Image.open(adir / (jp.stem + ".png")).convert("L").resize((W, H), Image.NEAREST)
        frames.append(np.asarray(img, dtype=np.uint8))
        masks.append(np.asarray(m) > 127)
    frames, masks = np.stack(frames), np.stack(masks)
    np.savez_compressed(f, frames=frames, masks=masks)
    return frames, masks


def labelled_indices(n_frames: int, per_video: int) -> list[int]:
    """Evenly spaced labelled frames, never the very last (it has no k+1 neighbour)."""
    if per_video >= n_frames - 1:
        return list(range(n_frames - 1))
    pos = np.linspace(0, n_frames - 2, per_video + 2)[1:-1] if per_video > 1 else np.array([(n_frames - 2) // 2])
    return sorted(set(int(round(p)) for p in pos))


BUDGETS = [1, 2, 5, 20]


def all_labelled_frames(train_videos: list[str]) -> dict[str, set[int]]:
    """Union over budgets of the labelled frames per video (what the offline generator must cover)."""
    out = {}
    for v in train_videos:
        n = len(load_video(v)[1])
        out[v] = set()
        for b in BUDGETS:
            out[v].update(labelled_indices(n, b))
    return out


def to_gray(frame_u8: np.ndarray) -> np.ndarray:
    return (frame_u8.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0


if __name__ == "__main__":
    tr, va = videos("train"), videos("val")
    print(f"train {len(tr)} videos, val {len(va)} videos")
    n = 0
    for v in tr + va:
        fr, mk = load_video(v)
        n += len(fr)
    print(f"cached {n} frames at {H}x{W}")
