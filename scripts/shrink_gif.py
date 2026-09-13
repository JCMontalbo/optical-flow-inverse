"""Shrink GIFs for the README without shrinking the picture: fewer colours, optional frame stride.

    python scripts/shrink_gif.py figures/*.gif --colors 96 --max-mb 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

RESERVED = np.array([(255, 0, 0), (0, 191, 191), (255, 127, 14), (31, 119, 180), (214, 39, 40), (0, 255, 255), (44, 160, 44)], float)


def snap_colours(frame: Image.Image) -> Image.Image:
    """Pixels with real chroma (contour lines, anti-aliased or not) snap to the nearest reserved colour."""
    a = np.asarray(frame).astype(float)
    chroma = a.max(-1) - a.min(-1)
    m = chroma > 40
    if m.any():
        d = ((a[m][:, None, :] - RESERVED[None]) ** 2).sum(-1)
        a[m] = RESERVED[d.argmin(1)]
    return Image.fromarray(a.astype(np.uint8))


def gray_palette():
    """100 grey levels plus the contour colours used in the figures: for grayscale GIFs with coloured lines,
    where an adaptive palette would drop the lines."""
    levels = [int(round(255 * i / 99)) for i in range(100)]
    cols = [(g, g, g) for g in levels] + [tuple(int(c) for c in rgb) for rgb in RESERVED]
    cols += [(0, 0, 0)] * (256 - len(cols))
    pal = Image.new("P", (1, 1))
    pal.putpalette([c for rgb in cols for c in rgb])
    return pal


def shrink(path: Path, colors: int, stride: int, max_mb: float, gray: bool = False):
    with Image.open(path) as im:
        dur = im.info.get("duration", 100)
        frames = [f.convert("RGB") for i, f in enumerate(ImageSequence.Iterator(im)) if i % stride == 0]
    before = path.stat().st_size / 1e6
    pal = gray_palette() if gray else None
    for c in (colors, 64, 48, 32):
        if pal is not None:
            q = [snap_colours(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
        else:
            q = [f.quantize(colors=c, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
        tmp = path.with_suffix(".tmp.gif")
        q[0].save(tmp, save_all=True, append_images=q[1:], duration=dur * stride, loop=0, optimize=True, disposal=1)
        if tmp.stat().st_size / 1e6 <= max_mb or c == 32 or pal is not None:
            break
    tmp.replace(path)
    print(f"{path.name}: {before:.1f} MB -> {path.stat().st_size / 1e6:.1f} MB ({len(frames)} frames, {c} colours, {q[0].size[0]}x{q[0].size[1]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gifs", nargs="+", type=Path)
    ap.add_argument("--colors", type=int, default=96)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-mb", type=float, default=6.0)
    ap.add_argument("--gray", action="store_true", help="fixed grey palette plus contour colours (for grayscale GIFs with coloured lines)")
    a = ap.parse_args()
    for g in a.gifs:
        shrink(g, a.colors, a.stride, a.max_mb, a.gray)
