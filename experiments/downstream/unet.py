"""A small U-Net (4 levels, 16-32-64-128, ~0.5M parameters) for binary segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, cin=3, widths=(16, 32, 64, 128)):
        super().__init__()
        self.enc = nn.ModuleList()
        c = cin
        for w in widths:
            self.enc.append(block(c, w))
            c = w
        self.bottom = block(widths[-1], widths[-1] * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        c = widths[-1] * 2
        for w in reversed(widths):
            self.up.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.dec.append(block(2 * w, w))
            c = w
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottom(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, s], 1))
        return self.head(x)


def bce_dice(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


if __name__ == "__main__":
    m = UNet()
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")
    print(m(torch.zeros(2, 3, 256, 448)).shape)
