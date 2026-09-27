"""DART-inspired skip CVAE and actor-conditioned latent Transformer."""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .geometry import DIM


def position(length, width, device, dtype):
    t = torch.arange(length, device=device, dtype=dtype)[:, None]
    f = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype) * (-math.log(10000) / width))
    p = torch.zeros(length, width, device=device, dtype=dtype)
    p[:, 0::2], p[:, 1::2] = torch.sin(t * f), torch.cos(t * f)
    return p[None]


class SkipEncoder(nn.Module):
    """Symmetric skip encoder pattern adapted from DART / DETR."""

    def __init__(self, width, layers, heads, ff_size, dropout):
        super().__init__()
        block = nn.TransformerEncoderLayer(width, heads, ff_size, dropout,
                                          activation="gelu", batch_first=True)
        n = (layers - 1) // 2
        self.down = nn.ModuleList([copy.deepcopy(block) for _ in range(n)])
        self.middle = copy.deepcopy(block)
        self.up = nn.ModuleList([copy.deepcopy(block) for _ in range(n)])
        self.skip = nn.ModuleList([nn.Linear(width * 2, width) for _ in range(n)])
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
        x = self.middle(x)
        for block, projection in zip(self.up, self.skip):
            x = block(projection(torch.cat((x, skips.pop()), -1)))
        return self.norm(x)


class HistoryTokens(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.actor = nn.Linear(DIM, width)
        self.reactor = nn.Linear(DIM, width)
        self.roles = nn.Parameter(torch.randn(2, 1, width) * 0.02)

    def forward(self, actor, reactor):
        a, r = self.actor(actor), self.reactor(reactor)
        p = position(a.shape[1], a.shape[2], a.device, a.dtype)
        return torch.cat((a + p + self.roles[0], r + p + self.roles[1]), 1)


class ReactionVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m, d = cfg["model"], cfg["data"]
        w, z = m["vae_hidden"], m["latent_dim"]
        self.future = d["future"]
        self.history = HistoryTokens(w)
        self.future_embed = nn.Linear(DIM, w)
        self.future_role = nn.Parameter(torch.randn(1, 1, w) * 0.02)
        self.distribution_tokens = nn.Parameter(torch.randn(1, 2, w) * 0.02)
        self.encoder = SkipEncoder(w, m["vae_layers"], m["heads"], m["ff_size"], m["dropout"])
        self.decoder = SkipEncoder(w, m["vae_layers"], m["heads"], m["ff_size"], m["dropout"])
        self.to_latent = nn.Linear(w, z)
        self.from_latent = nn.Linear(z, w)
        self.queries = nn.Parameter(torch.randn(1, d["future"], w) * 0.02)
        self.output = nn.Linear(w, DIM)
        self.register_buffer("latent_scale", torch.tensor(1.0))

    def _encode_distribution(self, actor, reactor, future):
        h = self.history(actor, reactor)
        f = self.future_embed(future)
        f = f + self.future_role + position(f.shape[1], f.shape[2], f.device, f.dtype)
        tokens = torch.cat((self.distribution_tokens.expand(len(h), -1, -1), h, f), 1)
        params = self.to_latent(self.encoder(tokens)[:, :2])
        mu, logvar = params[:, :1], params[:, 1:2].clamp(-10, 10)
        return mu, logvar

    def encode_mean(self, actor, reactor, future):
        """Deterministic posterior mean in eval mode; never sample or consume RNG."""
        if self.training:
            raise RuntimeError("encode_mean requires eval mode (encoder dropout must be disabled)")
        return self._encode_distribution(actor, reactor, future)[0]

    def encode(self, actor, reactor, future):
        mu, logvar = self._encode_distribution(actor, reactor, future)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return z, mu, logvar

    def decode(self, z, actor, reactor):
        h = self.history(actor, reactor)
        tokens = torch.cat((self.from_latent(z), h, self.queries.expand(len(h), -1, -1)), 1)
        return self.output(self.decoder(tokens)[:, -self.future:])


class ReactionDenoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        w = m["denoiser_hidden"]
        self.width, self.text_dropout = w, m["text_dropout"]
        self.history = HistoryTokens(w)
        self.text = nn.Linear(512, w)
        self.noise = nn.Linear(m["latent_dim"], w)
        self.time = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
        block = nn.TransformerEncoderLayer(w, m["heads"], m["ff_size"], m["dropout"],
                                          activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(block, m["denoiser_layers"], nn.LayerNorm(w))
        self.output = nn.Linear(w, m["latent_dim"])

    def forward(self, zt, step, actor, reactor, text, force_no_text=False):
        if force_no_text:
            text = torch.zeros_like(text)
        elif self.training and self.text_dropout:
            text = text * (torch.rand(len(text), 1, device=text.device) >= self.text_dropout)
        time = position(int(step.max()) + 1, self.width, zt.device, zt.dtype)[0][step]
        tokens = torch.cat((self.time(time)[:, None], self.text(text)[:, None],
                            self.history(actor, reactor), self.noise(zt)), 1)
        return self.output(self.transformer(tokens)[:, -1:])
