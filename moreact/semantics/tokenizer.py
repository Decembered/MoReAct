from __future__ import annotations

import copy

import numpy as np
import torch

from ..data import condition_window
from ..geometry import transform_features
from ..models import ReactionVAE
from ..train import load_checkpoint, save_checkpoint
from .common import VERSION, tensor_digest, require_identity


def windows(length, history, future):
    """(target start, exclusive end, newly covered frames), with overlapping tail."""
    if length < history + future:
        raise ValueError("Need at least history + future frames")
    starts = list(range(history, length - future + 1, future))
    result = [(s, s + future, future) for s in starts]
    if result[-1][1] < length:
        result.append((length - future, length, length - result[-1][1]))
    return np.asarray(result, dtype=np.int64)


class SharedMotionTokenizer:
    def __init__(self, vae, cfg, stats, data_digest):
        if vae.training or any(p.requires_grad for p in vae.parameters()):
            raise ValueError("Shared CVAE must already be frozen and in eval mode")
        if next(vae.parameters()).dtype != torch.float32:
            raise ValueError("Semantic encoding requires FP32 CVAE")
        self.vae = vae
        self.device = next(vae.parameters()).device
        self.cfg = copy.deepcopy(cfg)
        self.stats = {k: v.detach().clone().to(self.device) for k, v in stats.items()}
        self.data_digest = data_digest
        self.identity = self.current_identity()
        self._versions = self.versions()

    def versions(self):
        return [(id(x), x._version) for x in list(self.vae.parameters()) + list(self.vae.buffers()) + list(self.stats.values())]

    def current_identity(self):
        return dict(version=VERSION, vae=tensor_digest(self.vae.state_dict()),
                    stats=tensor_digest(self.stats), data_digest=self.data_digest,
                    fps=self.cfg["data"]["fps"], history=self.cfg["data"]["history"],
                    future=self.cfg["data"]["future"], latent_dim=self.cfg["model"]["latent_dim"],
                    feature_dim=276, roles=["actor", "reactor"])

    def assert_frozen(self, full=False):
        if self.vae.training or any(p.requires_grad for p in self.vae.parameters()):
            raise ValueError("Shared CVAE is no longer frozen/eval")
        if self.versions() != self._versions:
            raise ValueError("Shared CVAE/statistics mutated")
        if full:
            require_identity(self.identity, self.current_identity())

    @classmethod
    def from_generator(cls, generator):
        return cls(generator.vae, generator.cfg, generator.stats, generator.data_digest)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        saved = load_checkpoint(path)
        if saved["kind"] not in ("diffusion", "semantic_vae"):
            raise ValueError("Use a diffusion checkpoint or its exported semantic CVAE snapshot")
        # Constructing linear layers must not disturb the caller's sampling stream.
        with torch.random.fork_rng(devices=[]):
            vae = ReactionVAE(saved["config"])
        vae.load_state_dict(saved["vae"], strict=True)
        vae.to(device).eval().requires_grad_(False)
        result = cls(vae, saved["config"], saved["stats"], saved["data_digest"])
        if "identity" in saved:
            require_identity(saved["identity"], result.identity)
        return result

    def save_snapshot(self, path):
        self.assert_frozen(full=True)
        save_checkpoint(path, dict(kind="semantic_vae", vae=self.vae.state_dict(),
                                  config=self.cfg, stats=self.stats, data_digest=self.data_digest,
                                  identity=self.identity))

    def inputs(self, features, offsets, spans, role):
        """Target role is placed in the reactor slot, including reference/offsets."""
        h = self.identity["history"]
        target_hist = torch.stack([features[role, s-h:s] for s, _, _ in spans])
        partner_hist = torch.stack([features[1-role, s-h:s] for s, _, _ in spans])
        off = offsets[[1-role, role]][None].expand(len(spans), -1, -1)
        a, r, origin, basis = condition_window(partner_hist, target_hist, off, self.stats)
        target = torch.stack([features[role, s:e] for s, e, _ in spans])
        local = transform_features(target, origin, basis, off[:, 1])
        normalized = (local - self.stats["mean"]) / self.stats["std"]
        return a, r, normalized, origin, basis, target_hist

    @torch.no_grad()
    def encode_pair(self, features, betas, genders, offsets, batch_size=64, rvq=None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.assert_frozen()
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        off = torch.as_tensor(offsets, dtype=torch.float32, device=self.device)
        if x.ndim != 3 or x.shape[0] != 2 or x.shape[2] != 276:
            raise ValueError("Expected actor/reactor features [2,T,276]")
        if tuple(np.shape(betas)) != (2, 10) or tuple(np.shape(genders)) != (2,) or off.shape != (2, 3):
            raise ValueError("Invalid paired body parameters")
        if not torch.isfinite(x).all() or not torch.isfinite(off).all():
            raise ValueError("Nonfinite motion/body offsets")
        if not torch.isfinite(torch.as_tensor(betas)).all() or not torch.isin(torch.as_tensor(genders), torch.tensor([0, 1, 2], device=torch.as_tensor(genders).device)).all():
            raise ValueError("Invalid betas or gender codes")
        spans = windows(x.shape[1], self.identity["history"], self.identity["future"])
        values = []
        with torch.autocast(device_type=self.device.type, enabled=False):
            for role in range(2):
                chunks = []
                for start in range(0, len(spans), batch_size):
                    a, r, target, _, _, _ = self.inputs(x, off, spans[start:start+batch_size], role)
                    chunks.append(self.vae.encode_mean(a, r, target)[:, 0].cpu())
                values.append(torch.cat(chunks))
        mu = torch.stack(values, 1)
        if not torch.isfinite(mu).all():
            raise FloatingPointError("Nonfinite semantic latent")
        result = dict(mu=mu.numpy(), spans=spans, frames=np.asarray(x.shape[1]),
                      fps=np.asarray(self.identity["fps"]))
        if rvq is not None:
            require_identity(self.identity, getattr(rvq, "identity", None), "RVQ/CVAE")
            if rvq.training:
                raise ValueError("Token extraction requires frozen/eval RVQ")
            result["ids"] = rvq.ids(mu.to(rvq.codebooks.device)).cpu().numpy()
        return result
