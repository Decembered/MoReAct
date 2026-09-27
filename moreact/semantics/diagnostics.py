"""Posterior-mean/quantized reconstruction diagnostics, never generation metrics."""
from pathlib import Path

import numpy as np
import torch

from ..evaluate import motion_metrics
from ..geometry import BodyModels, transform_features
from .common import require_identity, sha256, load_arrays
from .tokenizer import SharedMotionTokenizer
from .rvq import load_rvq


@torch.no_grad()
def reconstruction_report(checkpoint, rvq, motion_cache, manifest, split="val", limit=8, device="cpu"):
    tokenizer = SharedMotionTokenizer.from_checkpoint(checkpoint, device)
    require_identity(tokenizer.identity, manifest["identity"])
    quantizer, ck = load_rvq(rvq, device, tokenizer.identity)
    require_identity(ck["rvq_id"], manifest["rvq_id"], "RVQ")
    body = BodyModels(tokenizer.cfg["data"]["body_models"], device)
    rows = [r for r in manifest["records"] if r["split"] == split]
    if limit:
        rows = rows[:limit]
    reports = []
    h, f = tokenizer.identity["history"], tokenizer.identity["future"]
    for row in rows:
        source = Path(motion_cache) / row["source_file"]
        require_identity(row["source_sha256"], sha256(source), "source motion")
        data = load_arrays(motion_cache, dict(file=row["source_file"]))
        encoded = tokenizer.encode_pair(data["features"], data["betas"], data["genders"], data["offsets"], rvq=quantizer)
        x = torch.tensor(data["features"], device=device)
        offsets = torch.tensor(data["offsets"], device=device)
        for role, name in enumerate(("actor", "reactor")):
            predictions = {"continuous_mu": [], "rvq": []}
            errors = {k: [] for k in predictions}
            for index, span in enumerate(encoded["spans"]):
                a, r, target, origin, basis, history = tokenizer.inputs(x, offsets, [span], role)
                mu = torch.from_numpy(encoded["mu"][index, role]).to(device)[None, None]
                codes = torch.from_numpy(encoded["ids"][index, role]).to(device)[None, None]
                for mode, z in (("continuous_mu", mu), ("rvq", quantizer.decode_ids(codes))):
                    local = tokenizer.vae.decode(z, a, r)
                    take = int(span[2])
                    errors[mode].extend((local[:, -take:] - target[:, -take:]).square().mean(-1).flatten().cpu().tolist())
                    world = transform_features(local * tokenizer.stats["std"] + tokenizer.stats["mean"],
                                               origin, basis, offsets[role][None], inverse=True)
                    world = body.repair(world, history, torch.tensor(data["betas"][role][None], device=device),
                                        torch.tensor(data["genders"][role:role+1], device=device))
                    predictions[mode].append(world[0, -take:].cpu().numpy())
            for mode in predictions:
                pred = np.concatenate(predictions[mode])
                metrics = motion_metrics(pred, data["features"][role, h:], data["features"][1-role, h:],
                                         tokenizer.identity["fps"], f, data["features"][role, h-1])
                metrics["normalized_feature_mse"] = float(np.mean(errors[mode]))
                reports.append(dict(episode=row["episode"], role=name, mode=mode, metrics=metrics))
    tokenizer.assert_frozen(full=True)
    summary = {}
    for role in ("actor", "reactor"):
        summary[role] = {}
        for mode in ("continuous_mu", "rvq"):
            selected = [r["metrics"] for r in reports if r["role"] == role and r["mode"] == mode]
            if selected:
                summary[role][mode] = {k: float(np.mean([r[k] for r in selected if r[k] is not None]))
                                      if any(r[k] is not None for r in selected) else None for k in selected[0]}
    return dict(summary=summary, samples=reports, kind="teacher_history_reconstruction",
                notes=["Actor target-slot encoding is transfer use, not a symmetrically trained CVAE.",
                       "Quantized decoding is diagnostic only; production generation stays continuous."])
