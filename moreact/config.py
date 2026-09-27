from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None):
    with (ROOT / "configs/default.yaml").open() as f:
        cfg = yaml.safe_load(f)
    if path:
        with Path(path).open() as f:
            merge(cfg, yaml.safe_load(f) or {})
    for key in ("source", "cache", "body_models", "role_overrides", "clip_model"):
        p = Path(cfg["data"][key]).expanduser()
        cfg["data"][key] = str(p if p.is_absolute() else ROOT / p)
    validate(cfg)
    return cfg


def validate(cfg):
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    if t.get('val_sampling', 'sequential') not in ('sequential', 'stratified_action'):
        raise ValueError('Unknown val_sampling')
    if 'lr_schedule_start_step' in t:
        origin = t['lr_schedule_start_step']
        total = sum(t[k] for k in ('stage1_steps', 'stage2_steps', 'stage3_steps'))
        if isinstance(origin, bool) or not isinstance(origin, int) or not 0 <= origin < total:
            raise ValueError('lr_schedule_start_step must be an integer within the training schedule')
    if min(d["history"], d["future"], d["primitives"]) < 1:
        raise ValueError("History, future and primitives must be positive")
    if not isinstance(d.get("stride", 1), int) or d.get("stride", 1) < 1:
        raise ValueError("Sampling stride must be a positive integer")
    if d["raw_fps"] % d["fps"]:
        raise ValueError("Use an integer downsampling ratio for causal preparation")
    if m["vae_layers"] % 2 != 1:
        raise ValueError("Skip VAE requires an odd number of layers")
    for key in ("vae_hidden", "denoiser_hidden"):
        if m[key] % m["heads"]:
            raise ValueError("Hidden dimensions must be divisible by heads")
    if m["diffusion_steps"] < 2 or t["batch_size"] < 1:
        raise ValueError("Invalid diffusion steps or batch size")
    required_losses = {"feature_rec", "smpl_joints_rec", "joint_fk_consistency",
                       "transl_delta", "joints_delta", "orient_delta",
                       "bone_length", "foot_contact"}
    if set(cfg.get("loss", {})) != required_losses:
        raise ValueError("loss must define exactly: " + ", ".join(sorted(required_losses)))
    if t["kl_weight"] < 0 or any(value < 0 for value in cfg["loss"].values()):
        raise ValueError("Loss weights must be non-negative")
    for section, allowed in (
        ('history_augmentation', {'probability', 'std_m', 'max_m'}),
        ('diffusion_loss_options', {'feature_root_weight', 'distance_map_threshold_m', 'recompute_target_deltas'}),
    ):
        values = cfg.get(section, {})
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError('Invalid ' + section + ' options')
        for key, value in values.items():
            if key == 'recompute_target_deltas':
                if not isinstance(value, bool):
                    raise ValueError('recompute_target_deltas must be boolean')
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(section + '.' + key + ' must be finite numeric')
            if value < 0 or (key in ('max_m', 'feature_root_weight', 'distance_map_threshold_m') and value == 0):
                raise ValueError('Invalid ' + section + '.' + key)
            if key == 'probability' and value > 1:
                raise ValueError('history_augmentation.probability must be in [0, 1]')
    if "diffusion_loss" in cfg:
        weights = cfg["diffusion_loss"]
        allowed = required_losses | {"latent_mse", "joint_velocity", "root_orientation",
                                     "root_angular_velocity", "root_position", "distance_map", "joint_contact",
                                     "root_relative_translation", "root_relative_orientation"}
        if set(weights) - allowed or weights.get("latent_mse", 0) <= 0:
            raise ValueError("Invalid diffusion_loss terms or nonpositive latent_mse weight")
        if any(not isinstance(v, (int, float)) or not (0 <= v < float('inf')) for v in weights.values()):
            raise ValueError("Diffusion loss weights must be finite and non-negative")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def clone(cfg):
    return copy.deepcopy(cfg)
