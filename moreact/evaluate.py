from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import dump_json
from .data import condition_window
from .geometry import BodyModels, JOINTS, transform_features
from .models import ReactionVAE
from .train import load_checkpoint, seed_all


def motion_metrics(pred, target, actor, fps, future, initial=None):
    p = np.asarray(pred[:, JOINTS]).reshape(-1, 22, 3)
    q = np.asarray(target[:, JOINTS]).reshape(-1, 22, 3)
    a = np.asarray(actor[:, JOINTS]).reshape(-1, 22, 3)
    root_error = np.linalg.norm(p[:, 0] - q[:, 0], axis=-1)
    relative = (p - p[:, :1]) - (q - q[:, :1])
    contact = p[:-1, [7, 8, 10, 11], 2] < 0.08
    foot_speed = np.linalg.norm(np.diff(p[:, [7, 8, 10, 11], :2], axis=0), axis=-1) * fps
    pair = np.linalg.norm(p[:, 0] - a[:, 0], axis=-1)
    true_pair = np.linalg.norm(q[:, 0] - a[:, 0], axis=-1)
    nearest = np.linalg.norm(p[:, :, None] - a[:, None], axis=-1).min(axis=(1, 2))
    positions = p
    if initial is not None:
        positions = np.concatenate((np.asarray(initial[JOINTS]).reshape(1, 22, 3), p), 0)
    velocity = np.diff(positions, axis=0) * fps
    jumps = []
    for index in range(future, len(p), future):
        # velocity[index] crosses a generation boundary when an initial frame is present.
        i = index if initial is not None else index - 1
        if 0 < i < len(velocity):
            jumps.append(np.linalg.norm(velocity[i] - velocity[i - 1], axis=-1).mean())
    return {"world_mpjpe_m": float(np.linalg.norm(p - q, axis=-1).mean()),
            "root_aligned_mpjpe_m": float(np.linalg.norm(relative, axis=-1).mean()),
            "root_ade_m": float(root_error.mean()), "root_fde_m": float(root_error[-1]),
            "pair_root_distance_m": float(pair.mean()),
            "pair_root_distance_error_m": float(np.abs(pair - true_pair).mean()),
            "nearest_interperson_joint_distance_m": float(nearest.mean()),
            "foot_sliding_mps": float(foot_speed[contact].mean()) if contact.any() else None,
            "foot_contact_fraction": float(contact.mean()) if contact.size else 0.,
            "boundary_velocity_jump_mps": float(np.mean(jumps)) if jumps else None,
            "jerk_mps3": float(np.linalg.norm(np.diff(positions, n=3, axis=0), axis=-1).mean() * fps**3)
            if len(positions) > 3 else None}


def aggregate(records):
    keys = {k for r in records for k in r["metrics"]}
    result = {}
    for key in sorted(keys):
        values = [r["metrics"][key] for r in records if r["metrics"][key] is not None]
        result[key] = {"mean": float(np.mean(values)) if values else None, "episodes": len(values)}
    return result


@torch.no_grad()
def reconstruct(checkpoint, cache, record, data, frames, device):
    cfg = checkpoint["config"]
    H, F = cfg["data"]["history"], cfg["data"]["future"]
    model = ReactionVAE(cfg).to(device).eval()
    model.load_state_dict(checkpoint["ema"])
    body = BodyModels(cfg["data"]["body_models"], device)
    stats = {k: v.to(device) for k, v in checkpoint["stats"].items()}
    offsets = torch.tensor(data["offsets"], device=device)[None]
    betas = torch.tensor(data["betas"], device=device)[None]
    genders = torch.tensor(data["genders"], device=device)[None]
    features = torch.tensor(data["features"], device=device)
    frames = min(frames, features.shape[1] - H)
    predictions, feature_errors = [], []
    for end in range(H, H + frames, F):
        ah, rh = features[0, end - H:end][None], features[1, end - H:end][None]
        a, r, origin, basis = condition_window(ah, rh, offsets, stats)
        target = features[1, end:min(end + F, H + frames)][None]
        take = target.shape[1]
        if take < F:
            target = torch.cat((target, target[:, -1:].expand(-1, F - take, -1)), 1)
        target_local = transform_features(target, origin, basis, offsets[:, 1])
        normalized = (target_local - stats["mean"]) / stats["std"]
        _, mu, _ = model.encode(a, r, normalized)
        rec = model.decode(mu, a, r)
        feature_errors.append(float((rec[:, :take] - normalized[:, :take]).square().mean()))
        world = transform_features(rec * stats["std"] + stats["mean"], origin, basis, offsets[:, 1], inverse=True)
        world = body.repair(world, rh, betas[:, 1], genders[:, 1])
        predictions.append(world[0, :take].cpu().numpy())
    pred = np.concatenate(predictions)
    metrics = motion_metrics(pred, data["features"][1, H:H + frames], data["features"][0, H:H + frames],
                             cfg["data"]["fps"], F, data["features"][1, H - 1])
    metrics["normalized_feature_mse"] = float(np.mean(feature_errors))
    return metrics


def evaluate(checkpoint, output, split="val", limit=8, frames=120, device="cpu", seed=0,
             ablations=True, video=False, cache=None):
    from .generate import load_episode, rollout
    saved = load_checkpoint(checkpoint)
    cfg = saved["config"]
    torch.set_num_threads(cfg["train"]["threads"])
    seed_all(seed)
    cache = cache or cfg["data"]["cache"]
    manifest = json.loads((Path(cache) / "manifest.json").read_text())
    if saved["data_digest"] != manifest["digest"]:
        raise ValueError("Checkpoint/data mismatch")
    records = [r for r in manifest["records"] if r["split"] == split and r["frames"] > cfg["data"]["history"]]
    if limit:
        records = records[:limit]
    if not records:
        raise ValueError("Empty evaluation split")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {"kind": saved["kind"], "checkpoint_step": saved["step"], "split": split,
              "data_digest": manifest["digest"], "seed": seed, "groups": {},
              "notes": ["Prediction errors are reference metrics, not complete generative quality scores.",
                        "Foot sliding uses predicted feet below 8 cm above the original common floor.",
                        "Nearest joint distance is a proximity proxy, not mesh penetration or contact accuracy.",
                        "shuffle means reversing the temporal order of observed actor frames.",
                        "VAE reconstruction uses ground-truth future and posterior mean; diffusion does not."]}
    if saved["kind"] == "vae":
        entries = []
        for record in records:
            _, data, _ = load_episode(cache, record["episode"])
            metrics = reconstruct(saved, cache, record, data, frames, device)
            entries.append({"episode": record["episode"], "metrics": metrics})
        result["groups"]["posterior_reconstruction"] = {"episodes": entries, "aggregate": aggregate(entries)}
    else:
        modes = [(False, "normal")]
        if ablations:
            modes = [(no_text, actor_mode) for no_text in (False, True)
                     for actor_mode in ("normal", "shuffle", "remove")]
        for no_text, actor_mode in modes:
            name = ("no_text" if no_text else "text") + "_actor_" + actor_mode
            entries = []
            for record in records:
                report = rollout(checkpoint, output / name / record["episode"], cache,
                                 record["episode"], split, frames, device, seed,
                                 no_text=no_text, actor_mode=actor_mode, video=video)
                entries.append({"episode": record["episode"], "metrics": report["metrics"],
                                "mean_segment_seconds": report["mean_segment_seconds"]})
            result["groups"][name] = {"episodes": entries, "aggregate": aggregate(entries),
                                      "mean_segment_seconds": float(np.mean([x["mean_segment_seconds"] for x in entries]))}
        baseline_dir = output / "text_actor_normal"
        # Paired identical-noise sensitivity; a changed output alone does not establish useful conditioning.
        for name, group in result["groups"].items():
            distances = []
            for record in records:
                with np.load(baseline_dir / record["episode"] / "motion.npz") as b:
                    baseline = b["reactor"][:, JOINTS].reshape(-1, 22, 3)
                with np.load(output / name / record["episode"] / "motion.npz") as b:
                    generated = b["reactor"][:, JOINTS].reshape(-1, 22, 3)
                distances.append(float(np.linalg.norm(generated - baseline, axis=-1).mean()))
            group["paired_output_change_m"] = float(np.mean(distances))
    dump_json(output / "report.json", result)
    print("Evaluation report: " + str(output / "report.json"), flush=True)
    return result
