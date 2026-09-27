from __future__ import annotations

import copy
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate

from .config import dump_json
from .augmentation import translate_history, repair_target_boundary
from .data import InterXDataset, condition_window, load_stats
from .diffusion import LatentDiffusion
from .geometry import BodyModels, transform_features
from .losses import vae_reconstruction_losses, diffusion_motion_losses
from .models import ReactionDenoiser, ReactionVAE
from .validation import audit_splits, validation_indices


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


@contextmanager
def preserved_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def load_checkpoint(path):
    # Only load trusted local checkpoints. Explicitly retain Python RNG state.
    return torch.load(path, map_location="cpu", weights_only=False)


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def rollout_probability(step, train_cfg):
    return min(1.0, max(0.0, (step - train_cfg["stage1_steps"]) /
                        max(train_cfg["stage2_steps"], 1)))


def learning_rate_at(step, train_cfg):
    """Shared single-device/DDP schedule, constant before an optional restart."""
    total = sum(train_cfg[k] for k in ('stage1_steps', 'stage2_steps', 'stage3_steps'))
    origin = train_cfg.get('lr_schedule_start_step', 0)
    if not 0 <= origin < total:
        raise ValueError('lr_schedule_start_step must be within the training schedule')
    fraction = min(1., max(0., (step - origin) / (total - origin)))
    return train_cfg['learning_rate'] * (1. - fraction)


def batch_at(dataset, indices, device):
    return {k: v.to(device) for k, v in default_collate([dataset[int(i)] for i in indices]).items()}


def compatible(a, b):
    for section, keys in {"data": ("history", "future", "fps"),
                          "model": ("latent_dim", "vae_hidden", "vae_layers", "heads", "ff_size", "dropout")}.items():
        for key in keys:
            if a[section][key] != b[section][key]:
                raise ValueError("VAE/config mismatch: " + section + "." + key)


class Trainer:
    def __init__(self, cfg, kind, output, vae_path=None, resume=None, overfit=False, resume_overrides=(), new_objective=False):
        if new_objective and (kind != 'diffusion' or not resume or (Path(output)/'metrics.jsonl').exists()):
            raise ValueError('New objective requires diffusion resume into a new run')
        self.cfg, self.kind = cfg, kind
        self.device = torch.device(cfg["train"]["device"])
        torch.set_num_threads(cfg["train"]["threads"])
        seed_all(cfg["train"]["seed"])
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "last.pt").exists() and not resume:
            raise ValueError("Run exists; use --resume or a new --output")
        self.dataset = InterXDataset(cfg, "train")
        self.validation = InterXDataset(cfg, "val")
        if cfg['train'].get('val_sampling', 'sequential') == 'stratified_action':
            dump_json(self.output / 'validation_split_audit.json', audit_splits(self.dataset, self.validation))
        self.stats = load_stats(cfg["data"]["cache"], self.device)
        self.body = BodyModels(cfg["data"]["body_models"], self.device)
        self.diffusion = LatentDiffusion(cfg["model"]["diffusion_steps"]).to(self.device)
        self.overfit = overfit
        self.step = 0
        saved = load_checkpoint(resume) if resume else None
        self.best_val = saved.get('best_val', float('inf')) if saved else float('inf')
        self.best_rollout = saved.get('best_rollout', float('inf')) if saved else float('inf')
        self.vae = ReactionVAE(cfg).to(self.device)
        if kind == "diffusion":
            if saved:
                self.vae.load_state_dict(saved["vae"])
            else:
                if not vae_path:
                    raise ValueError("train_diffusion requires --vae")
                ckpt = load_checkpoint(vae_path)
                if ckpt["kind"] != "vae":
                    raise ValueError("Expected a VAE checkpoint")
                compatible(cfg, ckpt["config"])
                if ckpt["data_digest"] != self.dataset.manifest["digest"]:
                    raise ValueError("VAE was trained on a different data cache")
                self.check_stats(ckpt)
                self.vae.load_state_dict(ckpt["ema"])
            self.vae.eval().requires_grad_(False)
            self.model = ReactionDenoiser(cfg).to(self.device)
            if not saved:
                self.estimate_scale()
        else:
            self.model = self.vae
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=0.)
        if saved:
            if saved["kind"] != kind or saved["data_digest"] != self.dataset.manifest["digest"]:
                raise ValueError("Checkpoint kind/data mismatch")
            if kind == 'diffusion' and not new_objective and cfg.get('diffusion_loss', {'latent_mse': 1.}) != saved['config'].get('diffusion_loss', {'latent_mse': 1.}):
                raise ValueError("Resume diffusion_loss mismatch; use a new run for a new objective")
            if kind == 'diffusion' and not new_objective:
                for key in ('diffusion_loss_options', 'history_augmentation'):
                    if cfg.get(key, {}) != saved['config'].get(key, {}):
                        raise ValueError('Resume ' + key + ' mismatch; use a new run for a new objective')
            self.check_stats(saved)
            compatible(cfg, saved["config"])
            for section in ("model", "data", "train", "loss"):
                if section not in saved["config"]:
                    raise ValueError("Resume checkpoint predates configured " + section + " settings")
                for key, value in cfg[section].items():
                    if section == 'train' and key in resume_overrides:
                        continue
                    if new_objective and section == 'data' and key == 'primitives':
                        continue
                    if key in ("device", "threads", "log_interval", "save_interval", "val_interval", "val_batches", "val_sampling"):
                        continue
                    if value != saved["config"][section].get(key):
                        raise ValueError("Resume config mismatch: " + section + "." + key)
            if saved["overfit"] != overfit:
                raise ValueError("Resume must preserve --overfit")
            self.model.load_state_dict(saved["model"])
            self.ema.load_state_dict(saved["ema"])
            self.optimizer.load_state_dict(saved["optimizer"])
            self.step = saved["step"]
            restore_rng(saved["rng"])
            if new_objective or cfg['train'].get('val_sampling', 'sequential') != saved['config']['train'].get('val_sampling', 'sequential'):
                self.best_val = self.best_rollout = float('inf')
        dump_json(self.output / "config.json", cfg)
        dump_json(self.output / "provenance.json", {"kind": kind, "data_digest": self.dataset.manifest["digest"],
                   "parameters": sum(p.numel() for p in self.model.parameters()),
                   "vae_checkpoint": str(vae_path) if vae_path else None, "overfit": overfit})

    @torch.no_grad()
    def estimate_scale(self):
        values = []
        H, F = self.cfg["data"]["history"], self.cfg["data"]["future"]
        with preserved_rng():
            seed_all(self.cfg["train"]["seed"] + 731)
            for _ in range(self.cfg["train"]["latent_scale_batches"]):
                indices = torch.randint(len(self.dataset), (self.cfg["train"]["batch_size"],))
                b = batch_at(self.dataset, indices, self.device)
                a, r, origin, basis = condition_window(b["actor"][:, :H], b["reactor"][:, :H], b["offsets"], self.stats)
                target = transform_features(b["reactor"][:, H:H + F], origin, basis, b["offsets"][:, 1])
                target = (target - self.stats["mean"]) / self.stats["std"]
                z, _, _ = self.vae.encode(a, r, target)
                values.append(z.flatten())
            scale = torch.cat(values).std(unbiased=False).clamp_min(1e-6)
            self.vae.latent_scale.copy_(scale)
        print("Train latent scale:", float(scale), flush=True)

    def check_stats(self, saved):
        for key in ("mean", "std"):
            if not torch.equal(self.stats[key].cpu(), saved["stats"][key].cpu()):
                raise ValueError("Checkpoint normalization differs from cache; use its original statistics or retrain")

    def losses(self, batch, model, probability=0.):
        d = self.cfg["data"]
        H, F, N = d["history"], d["future"], d["primitives"]
        context = batch["reactor"].clone()
        terms = []
        for k in range(N):
            start, end = k * F, k * F + H
            ah = batch["actor"][:, start:end]
            rh = context[:, start:end]
            future = batch['reactor'][:, end:end + F]
            recompute_deltas = (self.kind == 'diffusion' and
                self.cfg.get('diffusion_loss_options', {}).get('recompute_target_deltas', False))
            if self.kind == 'diffusion' and model.training and torch.is_grad_enabled():
                augmentation = self.cfg.get('history_augmentation', {})
                if augmentation.get('probability', 0.) > 0:
                    rh, selected = translate_history(rh, **augmentation)
                    if not recompute_deltas:
                        future = repair_target_boundary(future, rh, selected)
            if recompute_deltas:
                # Apply consistently to clean, generated and augmented history,
                # including rollout validation, not only noise-selected samples.
                future = repair_target_boundary(future, rh,
                    torch.ones(len(rh), dtype=torch.bool, device=rh.device))
            a, r, origin, basis = condition_window(ah, rh, batch["offsets"], self.stats)
            gt = transform_features(future, origin, basis, batch["offsets"][:, 1])
            gt = (gt - self.stats["mean"]) / self.stats["std"]
            if self.kind == "vae":
                z, mu, logvar = model.encode(a, r, gt)
                pred = model.decode(z, a, r)
                kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).mean()
                loss, reconstruction = vae_reconstruction_losses(
                    pred, gt, r, self.stats["mean"], self.stats["std"], self.body,
                    batch["betas"][:, 1], batch["genders"][:, 1], self.cfg["loss"])
                loss = loss + self.cfg["train"]["kl_weight"] * kl
                terms.append({"loss": loss, "kl": kl, **reconstruction})
            else:
                with torch.no_grad():
                    z, _, _ = self.vae.encode(a, r, gt)
                    z = z / self.vae.latent_scale
                step = torch.randint(self.diffusion.steps, (len(z),), device=self.device)
                noisy = self.diffusion.corrupt(z, step, torch.randn_like(z))
                predicted_z = model(noisy, step, a, r, batch["text"])
                weights = self.cfg.get('diffusion_loss', {'latent_mse': 1.})
                latent_loss = reconstruction_mse(predicted_z, z)
                loss = weights['latent_mse'] * latent_loss
                motion_terms = {}
                if any(value > 0 for key, value in weights.items() if key != 'latent_mse'):
                    # Do not use no_grad: gradients must pass THROUGH the frozen decoder.
                    pred = self.vae.decode(predicted_z * self.vae.latent_scale, a, r)
                    motion_loss, motion_terms = diffusion_motion_losses(
                        pred, gt, r, self.stats['mean'], self.stats['std'], self.body,
                        batch['betas'][:, 1], batch['genders'][:, 1], weights,
                        actor_future=transform_features(batch['actor'][:, end:end + F],
                            origin, basis, batch['offsets'][:, 0]).detach(),
                        options=self.cfg.get('diffusion_loss_options'),
                        target_history=(transform_features(
                            batch['reactor'][:, start:end], origin, basis, batch['offsets'][:, 1])
                            - self.stats['mean']) / self.stats['std']
                            if weights.get('root_relative_translation', 0) > 0 or
                               weights.get('root_relative_orientation', 0) > 0 else None)
                    loss = loss + motion_loss
                weighted = {'weighted_latent_mse': weights['latent_mse'] * latent_loss,
                            **{'weighted_' + key: weights[key] * value for key, value in motion_terms.items()}}
                terms.append({"loss": loss, "latent_mse": latent_loss, **motion_terms, **weighted})
            if k + 1 < N and probability > 0:
                mask = torch.rand(len(a), device=self.device) < probability
                if mask.any():
                    with torch.no_grad():
                        if self.kind == "diffusion":
                            was_training = model.training
                            model.eval()
                            sampled = self.diffusion.sample(model, a, r, batch["text"], self.cfg["model"]["latent_dim"])
                            model.train(was_training)
                            pred = self.vae.decode(sampled * self.vae.latent_scale, a, r)
                        local = pred.detach() * self.stats["std"] + self.stats["mean"]
                        world = transform_features(local, origin, basis, batch["offsets"][:, 1], inverse=True)
                        world = self.body.repair(world, rh, batch["betas"][:, 1], batch["genders"][:, 1])
                        context = context.clone()
                        context[:, end:end + F] = torch.where(mask[:, None, None], world, context[:, end:end + F])
        return {key: torch.stack([term[key] for term in terms]).mean() for key in terms[0]}

    @torch.no_grad()
    def validate(self):
        result = []
        rollout_result = []
        selection = validation_indices(self)
        with preserved_rng():
            seed_all(self.cfg["train"]["seed"] + 9001 +
                     (self.step if self.cfg['train'].get('val_sampling') == 'stratified_action' else 0))
            for i in range(self.cfg["train"]["val_batches"]):
                start = i * self.cfg['train']['batch_size']
                indices = selection[start:start + self.cfg['train']['batch_size']]
                batch = batch_at(self.validation, indices, self.device)
                result.append(self.losses(batch, self.ema))
                if self.cfg['train'].get('validate_rollout', False):
                    rollout_result.append(self.losses(batch, self.ema, 1.))
        output = {"val_" + key: float(torch.stack([x[key] for x in result]).mean()) for key in result[0]}
        if rollout_result:
            output.update({'val_rollout_' + key: float(torch.stack([x[key] for x in rollout_result]).mean())
                           for key in rollout_result[0]})
        return output

    def checkpoint(self, name='last.pt'):
        state = {"format_version": 1, "kind": self.kind, "step": self.step, "config": self.cfg,
                 "model": self.model.state_dict(), "ema": self.ema.state_dict(),
                 "optimizer": self.optimizer.state_dict(), "rng": rng_state(),
                 "stats": {k: v.cpu() for k, v in self.stats.items()},
                 "data_digest": self.dataset.manifest["digest"], "overfit": self.overfit,
                 "best_val": self.best_val, "best_rollout": self.best_rollout}
        if self.kind == "diffusion":
            state["vae"] = self.vae.state_dict()
        save_checkpoint(self.output / name, state)

    def run(self, steps=None):
        t = self.cfg["train"]
        total = sum(t[k] for k in ("stage1_steps", "stage2_steps", "stage3_steps"))
        stop = total if steps is None else steps
        if stop <= self.step or stop > total:
            raise ValueError("--steps is an absolute target and must exceed checkpoint step, within curriculum total")
        first = self.step
        start_time = time.perf_counter()
        self.model.train()
        with (self.output / "metrics.jsonl").open("a") as log:
            while self.step < stop:
                indices = (torch.arange(t["batch_size"]) % len(self.dataset) if self.overfit
                           else torch.randint(len(self.dataset), (t["batch_size"],)))
                batch = batch_at(self.dataset, indices, self.device)
                p = rollout_probability(self.step, t)
                for group in self.optimizer.param_groups:
                    group["lr"] = learning_rate_at(self.step, t)
                self.optimizer.zero_grad(set_to_none=True)
                losses = self.losses(batch, self.model, p)
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError("Nonfinite loss at step " + str(self.step))
                losses["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), t["grad_clip"], error_if_nonfinite=True)
                self.optimizer.step()
                with torch.no_grad():
                    for dst, src in zip(self.ema.parameters(), self.model.parameters()):
                        dst.lerp_(src, 1 - t["ema"])
                    for dst, src in zip(self.ema.buffers(), self.model.buffers()):
                        dst.copy_(src)
                self.step += 1
                record = {"step": self.step, "rollout_probability": p,
                          "learning_rate": self.optimizer.param_groups[0]['lr'],
                          **{k: float(v.detach()) for k, v in losses.items()}, "grad_norm": float(norm)}
                if self.step % t["val_interval"] == 0 or self.step == stop:
                    record.update(self.validate())
                    if record['val_loss'] < self.best_val:
                        self.best_val = record['val_loss']
                        self.checkpoint('best.pt')
                        dump_json(self.output / 'best.json', {'step': self.step, 'val_loss': self.best_val})
                    if record.get('val_rollout_loss', float('inf')) < self.best_rollout:
                        self.best_rollout = record['val_rollout_loss']
                        self.checkpoint('best_rollout.pt')
                        dump_json(self.output / 'best_rollout.json',
                                  {'step': self.step, 'val_rollout_loss': self.best_rollout})
                if (self.step % t["log_interval"] == 0 or self.step % t["val_interval"] == 0
                        or self.step == first + 1 or self.step == stop):
                    record["elapsed_seconds"] = time.perf_counter() - start_time
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)
                if self.step % t["save_interval"] == 0 or self.step == stop:
                    self.checkpoint()
        return record


def reconstruction_mse(pred, target):
    return F.mse_loss(pred, target)
