"""On-the-fly augmentation on the GPU: plausible affine, random elastic, and the recovered-flow family.

All geometric arms end in the same backward warp (image bilinear, mask nearest), so they differ only in
where the displacement field comes from:

* ``plausible_affine`` -- rotation +-10 deg, scale 0.9-1.1, translation +-6 px (no flips)
* ``elastic``          -- Gaussian noise smoothed with sigma 12 px, scaled to a random peak of 0-8 px
* ``flow_family``      -- a fresh member of the dissertation's family built on a recovered flow:
                          eps * flow, optionally gated by a Gaussian window with gain in [0, 3] (3.2),
                          optionally plus a windowed area-preserving generator with peak 0-8 px (4.5)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

GENERATORS = torch.tensor([[[0.0, -1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, -1.0]], [[0.0, 1.0], [0.0, 0.0]], [[1.0, 1.0], [3.0, -1.0]]])


def _grid(n, H, W, device):
    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    return xs.expand(n, H, W), ys.expand(n, H, W)


def warp(x, y, du, dv):
    """Backward warp by a displacement field in pixels: out(p) = in(p - d(p)). x: [n,1,H,W], du/dv: [n,H,W]."""
    n, _, H, W = x.shape
    xs, ys = _grid(n, H, W, x.device)
    gx = (xs - du) * 2 / (W - 1) - 1
    gy = (ys - dv) * 2 / (H - 1) - 1
    grid = torch.stack([gx, gy], -1)
    xo = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
    yo = F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=True)
    return xo, yo


def _u(n, lo, hi, gen, device):
    return (torch.rand(n, generator=gen) * (hi - lo) + lo).to(device)


def plausible_affine(x, y, gen):
    n, _, H, W = x.shape
    dev = x.device
    ang = _u(n, -10, 10, gen, dev) * math.pi / 180
    sc = _u(n, 0.9, 1.1, gen, dev)
    tx, ty = _u(n, -6, 6, gen, dev), _u(n, -6, 6, gen, dev)
    xs, ys = _grid(n, H, W, dev)
    cx, cy = (W - 1) / 2, (H - 1) / 2
    c, s = torch.cos(ang).view(-1, 1, 1), torch.sin(ang).view(-1, 1, 1)
    k = (1 / sc).view(-1, 1, 1)  # inverse map: where does output pixel p come from
    dx, dy = xs - cx, ys - cy
    sx = cx + k * (c * dx + s * dy) - tx.view(-1, 1, 1)
    sy = cy + k * (-s * dx + c * dy) - ty.view(-1, 1, 1)
    return warp(x, y, xs - sx, ys - sy)


def _gaussian_blur(z, sigma):
    r = int(3 * sigma)
    t = torch.arange(-r, r + 1, device=z.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    z = F.conv2d(F.pad(z, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(z.shape[1], 1, 1, -1), groups=z.shape[1])
    z = F.conv2d(F.pad(z, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(z.shape[1], 1, -1, 1), groups=z.shape[1])
    return z


def elastic(x, y, gen, sigma=12.0, max_peak=8.0):
    n, _, H, W = x.shape
    noise = torch.randn(n, 2, H, W, generator=gen).to(x.device)
    d = _gaussian_blur(noise, sigma)
    peak = d.flatten(2).norm(dim=1).amax(-1).clamp(min=1e-6)  # [n]
    target = _u(n, 0, max_peak, gen, x.device)
    d = d * (target / peak).view(-1, 1, 1, 1)
    return warp(x, y, d[:, 0], d[:, 1])


def _window(n, H, W, center, width, device):
    xs, ys = _grid(n, H, W, device)
    return torch.exp(-((xs - center[:, 0].view(-1, 1, 1)) ** 2 + (ys - center[:, 1].view(-1, 1, 1)) ** 2) / (2 * width.view(-1, 1, 1) ** 2))


def _stream_perturbation(n, H, W, center, width, A, device):
    """Windowed stream function of a traceless generator -> divergence-free field, unit peak."""
    xs, ys = _grid(n, H, W, device)
    dx, dy = xs - center[:, 0].view(-1, 1, 1), ys - center[:, 1].view(-1, 1, 1)
    a, b, c = A[:, 0, 0].view(-1, 1, 1), A[:, 0, 1].view(-1, 1, 1), A[:, 1, 0].view(-1, 1, 1)
    psi = _window(n, H, W, center, width, device) * (a * dx * dy + 0.5 * b * dy**2 - 0.5 * c * dx**2)
    psi_y, psi_x = torch.gradient(psi, dim=(1, 2))
    u, v = psi_y, -psi_x
    peak = torch.sqrt(u**2 + v**2).flatten(1).amax(1).clamp(min=1e-6).view(-1, 1, 1)
    return u / peak, v / peak


def flow_family(x, y, flow, centers, gen, max_peak=8.0):
    """flow: [n,2,H,W] recovered flow (zeros where none was accepted); centers: [n,2] a point on the atrium."""
    n, _, H, W = x.shape
    dev = x.device
    eps = _u(n, 0.25, 1.75, gen, dev).view(-1, 1, 1)
    du, dv = eps * flow[:, 0], eps * flow[:, 1]
    # 3.2: with p = 1/2 gate the motion with a Gaussian window near the atrium, gain in [0, 3]
    use_w = (torch.rand(n, generator=gen) < 0.5).to(dev).view(-1, 1, 1)
    cen = centers + (torch.rand(n, 2, generator=gen).to(dev) * 32 - 16)
    f = _window(n, H, W, cen, _u(n, 12, 32, gen, dev), dev)
    g = _u(n, 0, 3, gen, dev).view(-1, 1, 1)
    gate = torch.where(use_w, 1 + (g - 1) * f, torch.ones_like(f))
    du, dv = du * gate, dv * gate
    # 4.5: with p = 1/2 add a windowed area-preserving generator, peak 0-8 px, random sign
    use_p = (torch.rand(n, generator=gen) < 0.5).to(dev).view(-1, 1, 1)
    A = GENERATORS.to(dev)[torch.randint(0, 4, (n,), generator=gen).to(dev)]
    cen2 = centers + (torch.rand(n, 2, generator=gen).to(dev) * 32 - 16)
    pu, pv = _stream_perturbation(n, H, W, cen2, _u(n, 12, 32, gen, dev), A, dev)
    amp = (_u(n, 0, max_peak, gen, dev) * torch.where(torch.rand(n, generator=gen) < 0.5, -1.0, 1.0).to(dev)).view(-1, 1, 1)
    du = du + torch.where(use_p, amp * pu, torch.zeros_like(pu))
    dv = dv + torch.where(use_p, amp * pv, torch.zeros_like(pv))
    return warp(x, y, du, dv)


def intensity_jitter(x, gen):
    n = x.shape[0]
    dev = x.device
    gamma = _u(n, 0.8, 1.25, gen, dev).view(-1, 1, 1, 1)
    contrast = _u(n, 0.9, 1.1, gen, dev).view(-1, 1, 1, 1)
    noise = torch.randn(x.shape, generator=gen).to(dev) * 0.02
    x = x.clamp(0, 1) ** gamma
    x = (x - 0.5) * contrast + 0.5 + noise
    return x.clamp(0, 1)
