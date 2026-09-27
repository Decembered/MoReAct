from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..config import dump_json
from ..data import load_stats
from ..train import save_checkpoint, load_checkpoint
from .common import (VERSION, new_output, sha256, tensor_digest, require_identity,
                     load_arrays, read_manifest, write_manifest)
from .rvq import ResidualQuantizer, load_rvq
from .tokenizer import SharedMotionTokenizer


def cache_latents(checkpoint, cache, output, device="cpu", limit=0, batch_size=64):
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    checkpoint_id = sha256(checkpoint)
    tokenizer = SharedMotionTokenizer.from_checkpoint(checkpoint, device)
    if checkpoint_id != sha256(checkpoint):
        raise ValueError("Checkpoint changed during loading; use an immutable step checkpoint")
    cache = Path(cache)
    manifest = json.loads((cache / "manifest.json").read_text())
    require_identity(tokenizer.data_digest, manifest["digest"], "data")
    require_identity(tokenizer.identity["stats"], tensor_digest(load_stats(cache, "cpu")), "statistics")
    out = new_output(output)
    tokenizer.save_snapshot(out / "vae_snapshot.pt")
    records, counts, seen = [], {}, set()
    for record in manifest["records"]:
        ep, split = record["episode"], record["split"]
        if ep in seen or split not in ("train", "val", "test"):
            raise ValueError("Duplicate episode or invalid split in source")
        seen.add(ep)
        if limit and counts.get(split, 0) >= limit:
            continue
        data = load_arrays(cache, record)
        encoded = tokenizer.encode_pair(data["features"], data["betas"], data["genders"], data["offsets"], batch_size)
        captions = [str(s).strip() for s in data["captions"]]
        if not captions or not all(captions):
            raise ValueError("Missing interaction captions: " + ep)
        # Names are generated rather than accepting episode IDs as output paths.
        filename = "%06d.npz" % len(records)
        np.savez_compressed(out / filename, **encoded)
        records.append(dict(episode=ep, split=split, file=filename, sha256=sha256(out / filename),
                            captions=captions, frames=int(encoded["frames"]),
                            actor_id=record.get("actor_id", "actor"), reactor_id=record.get("reactor_id", "reactor"),
                            source_file=record["file"], source_sha256=sha256(cache / record["file"])))
        counts[split] = counts.get(split, 0) + 1
    if not records:
        raise ValueError("Empty source cache")
    tokenizer.assert_frozen(full=True)
    return write_manifest(out, dict(kind="latents", identity=tokenizer.identity, records=records,
                                   source_cache=str(cache.resolve()), counts=counts, limit_per_split=limit,
                                   source_checkpoint=str(Path(checkpoint).resolve()), source_checkpoint_sha256=checkpoint_id,
                                   snapshot_sha256=sha256(out / "vae_snapshot.pt")))


def latent_split(cache, manifest, split):
    values = [load_arrays(cache, r)["mu"] for r in manifest["records"] if r["split"] == split]
    if not values:
        raise ValueError("Missing " + split + " latent split")
    x = torch.from_numpy(np.concatenate(values))
    if x.ndim != 3 or x.shape[1:] != (2, manifest["identity"]["latent_dim"]) or not torch.isfinite(x).all():
        raise ValueError("Invalid paired latent cache")
    return x


@torch.no_grad()
def quantization_metrics(model, values, batch_size=1024):
    squared = torch.zeros(2, dtype=torch.float64)
    counts = torch.zeros(model.config["levels"], model.config["size"], dtype=torch.int64)
    total = 0
    for x in values.split(batch_size):
        ids = model.ids(x.to(model.codebooks.device))
        pred = model.decode_ids(ids).cpu()
        squared += (pred - x).double().square().sum((0, 2))
        total += len(x) * x.shape[-1]
        for level in range(model.config["levels"]):
            counts[level] += torch.bincount(ids[..., level].flatten().cpu(), minlength=model.config["size"])
    per_role = (squared / total).tolist()
    probs = counts.double() / counts.sum(-1, keepdim=True).clamp_min(1)
    return dict(actor_mse=per_role[0], reactor_mse=per_role[1], mean_mse=sum(per_role)/2,
                used_codes=(counts > 0).sum(-1).tolist(),
                perplexity=torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum(-1)).tolist())


def fit_rvq(cache, output, epochs=20, batch_size=1024, size=256, levels=2,
            decay=0.99, seed=0, device="cpu", resume=None):
    if min(epochs, batch_size) < 1:
        raise ValueError("epochs/batch_size must be positive")
    manifest = read_manifest(cache, "latents")
    train, val = latent_split(cache, manifest, "train"), latent_split(cache, manifest, "val")
    cfg = dict(dim=manifest["identity"]["latent_dim"], size=size, levels=levels, decay=decay)
    model = ResidualQuantizer(**cfg).to(device)
    rng = torch.Generator().manual_seed(seed)
    protocol = dict(batch_size=batch_size, seed=seed, cache_id=manifest["manifest_id"])
    start, best = 0, float("inf")
    out = Path(output)
    if resume:
        saved = load_checkpoint(resume)
        if saved.get("kind") != "semantic_rvq" or saved.get("version") != VERSION:
            raise ValueError("Invalid RVQ resume checkpoint")
        require_identity(saved["identity"], manifest["identity"])
        if saved["config"] != cfg or saved["protocol"] != protocol:
            raise ValueError("RVQ resume configuration/cache mismatch")
        model.load_state_dict(saved["model"], strict=True)
        require_identity(saved["rvq_id"], tensor_digest(model.state_dict()), "RVQ")
        rng.set_state(saved["rng"])
        start, best = saved["epoch"], saved["best"]
        if out.resolve() != Path(resume).resolve().parent:
            raise ValueError("Resume must use original RVQ output directory")
    else:
        new_output(out)
    if epochs <= start:
        raise ValueError("epochs is an absolute target beyond the saved epoch")
    for epoch in range(start, epochs):
        model.train()
        order = torch.randperm(len(train), generator=rng)
        for ids in order.split(batch_size):
            # Every selected window contributes both roles, hence equal population weight.
            model.update(train[ids].to(device))
        model.eval()
        metrics = quantization_metrics(model, val, batch_size)
        improved = metrics["mean_mse"] < best
        best = min(best, metrics["mean_mse"])
        saved = dict(kind="semantic_rvq", version=VERSION, identity=manifest["identity"],
                     config=cfg, model=model.state_dict(), rvq_id=tensor_digest(model.state_dict()),
                     protocol=protocol, epoch=epoch+1, best=best, rng=rng.get_state(), metrics=metrics)
        save_checkpoint(out / "last.pt", saved)
        if improved:
            save_checkpoint(out / "best.pt", saved)
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(epoch=epoch+1, **metrics), allow_nan=False) + "\n")
    return saved["metrics"]


def cache_tokens(cache, rvq, output, device="cpu"):
    manifest = read_manifest(cache, "latents")
    model, saved = load_rvq(rvq, device, manifest["identity"])
    out = new_output(output)
    records = []
    before = tensor_digest(model.state_dict())
    for record in manifest["records"]:
        data = load_arrays(cache, record)
        x = torch.from_numpy(data.pop("mu")).to(device)
        data["ids"] = model.ids(x).cpu().numpy()
        np.savez_compressed(out / record["file"], **data)
        records.append(dict(record, sha256=sha256(out / record["file"])))
    require_identity(before, tensor_digest(model.state_dict()), "RVQ frozen state")
    return write_manifest(out, dict(kind="tokens", identity=manifest["identity"],
                                   rvq_id=saved["rvq_id"], rvq_config=saved["config"],
                                   latent_cache_id=manifest["manifest_id"], records=records,
                                   counts=manifest["counts"], limit_per_split=manifest["limit_per_split"]))
