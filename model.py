#!/usr/bin/env python3
"""
dlss5-distill: Real-Time 2x Neural Super-Resolution Architecture
Copyright 2026 Benjamin Wegener. Licensed under Apache 2.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RepConv3x3Deployable(nn.Module):
    """
    Structural Re-Parameterization Block.
    - Training: 3x3 Conv + 1x1 Conv + Identity Skip Branch
    - Deploy: Mathematically fused into a single 3x3 Conv2D layer
    """
    def __init__(self, in_ch, out_ch, deploy=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.deploy = deploy

        if deploy:
            self.rbr = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True)
        else:
            self.rbr_dense = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True)
            self.rbr_1x1 = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
            self.rbr_id = nn.Identity() if in_ch == out_ch else None

        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        if self.deploy:
            return self.act(self.rbr(x))
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_id is not None:
            out = out + self.rbr_id(x)
        return self.act(out)

    def switch_to_deploy(self):
        if self.deploy:
            return
        k = self.rbr_dense.weight.data + F.pad(self.rbr_1x1.weight.data, [1, 1, 1, 1])
        b = self.rbr_dense.bias.data + self.rbr_1x1.bias.data
        if self.rbr_id is not None:
            for i in range(self.in_ch):
                k[i, i, 1, 1] += 1.0
        self.rbr = nn.Conv2d(self.in_ch, self.out_ch, 3, 1, 1, bias=True)
        self.rbr.weight.data = k
        self.rbr.bias.data = b
        del self.rbr_dense, self.rbr_1x1
        if hasattr(self, 'rbr_id'):
            del self.rbr_id
        self.deploy = True

class RepGANBlock(nn.Module):
    """Residual block with RepConv3x3 and parameter-free spatial self-attention."""
    def __init__(self, ch=28, deploy=False):
        super().__init__()
        self.conv1 = RepConv3x3Deployable(ch, ch, deploy=deploy)
        self.conv2 = RepConv3x3Deployable(ch, ch, deploy=deploy)
        self.attn = nn.Tanh()

    def forward(self, x):
        r = x
        feat = self.conv2(self.conv1(x))
        feat = feat * self.attn(feat)
        return r + feat

    def switch_to_deploy(self):
        self.conv1.switch_to_deploy()
        self.conv2.switch_to_deploy()

class Phase4Generator(nn.Module):
    """
    Compact real-time 2x super-resolution generator (~48k params fused).
    Combines deep residual trunk, sub-pixel upscaling, and global bicubic bypass.
    """
    def __init__(self, in_ch=3, out_ch=3, base_ch=28, num_blocks=4, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.head = RepConv3x3Deployable(in_ch, base_ch, deploy=deploy)
        self.blocks = nn.ModuleList([RepGANBlock(base_ch, deploy=deploy) for _ in range(num_blocks)])
        self.trunk = RepConv3x3Deployable(base_ch, base_ch, deploy=deploy)

        # 2x PixelShuffle Upscaler
        self.up = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(base_ch),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1)
        )

        self.skip_scale = nn.Parameter(torch.ones(1, out_ch, 1, 1))
        self.skip_bias  = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.res_gain   = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        bic = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        bic = bic * self.skip_scale + self.skip_bias

        f0 = self.head(x)
        feat = f0
        for b in self.blocks:
            feat = b(feat)
        t = self.trunk(feat)
        res = self.up(f0 + t) * self.res_gain

        return torch.clamp(bic + res, 0.0, 1.0)

    def switch_to_deploy(self):
        self.head.switch_to_deploy()
        for b in self.blocks:
            b.switch_to_deploy()
        self.trunk.switch_to_deploy()
        self.deploy = True

class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for texture and micro-edge assessment."""
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_ch * 4, 1, 3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)
