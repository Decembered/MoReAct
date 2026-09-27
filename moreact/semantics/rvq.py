"""FP32 EMA residual quantization extracted from ttr_remogen.semantic.vq.

Adapted from TTR/OpenMotionLab EMAReset + ResidualQuantizer; see TTR_LICENSE.
No encoder/decoder, reconstruction gradients, unit normalization or distributed state.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualQuantizer(nn.Module):
    def __init__(self, dim=128, size=256, levels=2, decay=0.99):
        super().__init__()
        if min(dim, size) < 1 or levels not in (1, 2) or not 0 < decay < 1:
            raise ValueError("Invalid RVQ configuration")
        self.config = dict(dim=dim, size=size, levels=levels, decay=decay)
        self.register_buffer("codebooks", torch.zeros(levels, size, dim))
        self.register_buffer("sums", torch.zeros(levels, size, dim))
        self.register_buffer("counts", torch.zeros(levels, size))
        self.register_buffer("initialized", torch.zeros(levels, dtype=torch.bool))

    def _input(self, x):
        if x.shape[-1] != self.config["dim"] or x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("RVQ requires finite FP32 latent vectors of the configured width")
        return x

    def nearest(self, x, level):
        c = self.codebooks[level]
        return (x.square().sum(-1, keepdim=True) - 2 * x @ c.T + c.square().sum(-1)).argmin(-1)

    @torch.no_grad()
    def ids(self, x):
        if not self.initialized.all():
            raise ValueError("RVQ has not been fitted")
        residual = self._input(x).clone()
        codes = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            for level in range(self.config["levels"]):
                code = self.nearest(residual, level)
                codes.append(code)
                residual -= F.embedding(code, self.codebooks[level])
        return torch.stack(codes, -1)

    def decode_ids(self, ids):
        if ids.shape[-1] != self.config["levels"] or ids.dtype != torch.long:
            raise ValueError("Expected int64 residual codes ending in levels")
        if (ids < 0).any() or (ids >= self.config["size"]).any():
            raise ValueError("Code out of range")
        return sum(F.embedding(ids[..., k], self.codebooks[k]) for k in range(self.config["levels"]))

    @torch.no_grad()
    def update(self, x):
        if not self.training:
            raise RuntimeError("Cannot update frozen RVQ")
        residual = self._input(x).reshape(-1, self.config["dim"]).clone()
        if not len(residual):
            raise ValueError("Empty RVQ training batch")
        decay, size = self.config["decay"], self.config["size"]
        with torch.autocast(device_type=x.device.type, enabled=False):
            for level in range(self.config["levels"]):
                # Deterministic train-batch replacements; caller shuffles using a private RNG.
                replacements = residual[torch.arange(size, device=x.device) % len(residual)]
                if not self.initialized[level]:
                    self.codebooks[level].copy_(replacements)
                    self.sums[level].copy_(replacements)
                    self.counts[level].fill_(1)
                    self.initialized[level] = True
                code = self.nearest(residual, level)
                quantized = F.embedding(code, self.codebooks[level]).clone()
                count = torch.bincount(code, minlength=size).to(residual.dtype)
                sums = torch.zeros_like(self.sums[level]).index_add_(0, code, residual)
                self.counts[level].mul_(decay).add_(count, alpha=1-decay)
                self.sums[level].mul_(decay).add_(sums, alpha=1-decay)
                centers = self.sums[level] / self.counts[level, :, None].clamp_min(1e-8)
                self.codebooks[level].copy_(torch.where(self.counts[level, :, None] >= 1, centers, replacements))
                residual -= quantized


def load_rvq(path, device="cpu", identity=None):
    from ..train import load_checkpoint
    from .common import VERSION, require_identity, tensor_digest
    ck = load_checkpoint(path)
    if ck.get("kind") != "semantic_rvq" or ck.get("version") != VERSION:
        raise ValueError("Not a shared-CVAE RVQ checkpoint")
    if identity is not None:
        require_identity(identity, ck["identity"])
    model = ResidualQuantizer(**ck["config"]).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().requires_grad_(False)
    if not model.initialized.all() or tensor_digest(model.state_dict()) != ck["rvq_id"]:
        raise ValueError("RVQ identity or initialization mismatch")
    model.identity = ck["identity"]
    model.rvq_id = ck["rvq_id"]
    return model, ck
