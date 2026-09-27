"""Cosine DDPM with x0 prediction, as used by DART; text-only CFG."""
from __future__ import annotations

import math
import torch
from torch import nn


class LatentDiffusion(nn.Module):
    def __init__(self, steps=10):
        super().__init__()
        self.steps = steps
        t = torch.arange(steps + 1, dtype=torch.float64) / steps
        abar = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
        beta = (1 - abar[1:] / abar[:-1]).clamp(max=0.999).float()
        alpha = 1 - beta
        abar = alpha.cumprod(0)
        prev = torch.cat((torch.ones(1), abar[:-1]))
        for name, value in {
            "abar": abar,
            "posterior_variance": beta * (1 - prev) / (1 - abar),
            "coef_x0": beta * prev.sqrt() / (1 - abar),
            "coef_xt": (1 - prev) * alpha.sqrt() / (1 - abar),
        }.items():
            self.register_buffer(name, value)

    def corrupt(self, z, step, noise):
        a = self.abar[step, None, None]
        return a.sqrt() * z + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, actor, reactor, text, latent_dim, guidance=1.0, generator=None):
        z = torch.randn(len(actor), 1, latent_dim, device=actor.device, generator=generator)
        for i in reversed(range(self.steps)):
            step = torch.full((len(actor),), i, device=actor.device, dtype=torch.long)
            x0 = model(z, step, actor, reactor, text)
            if guidance != 1.0:
                null = model(z, step, actor, reactor, text, force_no_text=True)
                x0 = null + guidance * (x0 - null)
            z = self.coef_x0[i] * x0 + self.coef_xt[i] * z
            if i:
                noise = torch.randn(z.shape, device=z.device, generator=generator)
                z = z + self.posterior_variance[i].sqrt() * noise
        return z
