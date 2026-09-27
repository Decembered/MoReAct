from __future__ import annotations

import json
import pickle
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import digest, dump_json
from .geometry import (BodyModels, GENDERS, make_features, reference_frame,
                       transform_features)

FEATURE_VERSION = "dart276_backward_deltas_shared_zup_v1"
STATS_VERSION = "shared_equal_roles_full_window_v1"


def read_splits(root):
    result = {}
    for split in ("train", "val", "test"):
        ids = (Path(root) / "splits" / (split + ".txt")).read_text().split()
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate episodes in " + split)
        result[split] = ids
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(result[a]) & set(result[b])
        if overlap:
            raise ValueError("Split leakage: " + str(sorted(overlap)[:5]))
    return result


def resolve_role(episode, order, overrides):
    if order not in (0, 1):
        raise ValueError("interaction_order must be 0 or 1: " + episode)
    if episode in overrides:
        item = overrides[episode]
        if item.get("status") != "motion_verified" or not item.get("evidence"):
            raise ValueError("Role override requires motion verification evidence")
        actor, status = item["actor"], "motion_verified"
        evidence = item["evidence"]
    else:
        actor, status = ("P1" if order == 0 else "P2"), "annotation_assumed"
        evidence = "Inter-X README: 0=P1 actor; 1=P2 actor. Not independently verified."
    if actor not in ("P1", "P2"):
        raise ValueError("Invalid actor ID")
    return {"actor_id": actor, "reactor_id": "P2" if actor == "P1" else "P1",
            "role_status": status, "role_evidence": evidence, "original_order": int(order)}


def cache_settings(cfg, limit):
    d = cfg["data"]
    return {"feature_version": FEATURE_VERSION, "source": d["source"],
            "fps": d["fps"], "raw_fps": d["raw_fps"], "history": d["history"],
            "future": d["future"], "body_models": d["body_models"],
            "role_overrides": json.loads(Path(d["role_overrides"]).read_text()),
            "clip_model": d["clip_model"], "limit_per_split": limit,
            "selection_seed": cfg["train"]["seed"]}


def prepare(cfg, limit=None, device="cpu", reuse_cache=None):
    from scipy.spatial.transform import Rotation
    from tqdm import tqdm
    from .text import FrozenCLIP

    d = cfg["data"]
    root, cache = Path(d["source"]), Path(d["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "episodes").mkdir(exist_ok=True)
    settings = cache_settings(cfg, limit)
    reuse_settings = None
    if reuse_cache is not None:
        reuse_cache = Path(reuse_cache)
        if reuse_cache.resolve() == cache.resolve():
            raise ValueError("Reuse cache must differ from destination")
        reuse_settings = json.loads((reuse_cache / "settings.json").read_text())
        # Episode features do not depend on window length or sample selection.
        ignored = {"history", "future", "limit_per_split", "selection_seed"}
        if {k: v for k, v in settings.items() if k not in ignored} != {
                k: v for k, v in reuse_settings.items() if k not in ignored}:
            raise ValueError("Reuse cache has incompatible feature/source settings")
    settings_path = cache / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Cache settings differ; select a new data.cache directory")
    dump_json(settings_path, settings)
    splits = read_splits(root)
    split_digest = digest(splits)
    with (root / "annots/interaction_order.pkl").open("rb") as f:
        orders = pickle.load(f)
    overrides = settings["role_overrides"]
    body = BodyModels(d["body_models"], device)
    clip = FrozenCLIP(d["clip_model"], device)
    rng = np.random.RandomState(cfg["train"]["seed"])
    manifest, skipped = [], []
    rotate_y_to_z = torch.tensor([[1., 0, 0], [0, 0, -1], [0, 1, 0]], device=device)
    with zipfile.ZipFile(root / "texts.zip") as archive:
        text_index = {Path(n).stem: n for n in archive.namelist() if n.endswith(".txt")}
        for split, all_ids in splits.items():
            ids = all_ids
            if limit is not None:
                ids = sorted(rng.choice(ids, min(limit, len(ids)), replace=False).tolist())
            for episode in tqdm(ids, desc="prepare " + split):
                role = resolve_role(episode, orders[episode], overrides)
                target = cache / "episodes" / (episode + ".npz")
                # Errors in physical samples are recorded, never silently relabeled.
                try:
                    texts = [s.strip() for s in archive.read(text_index[episode]).decode("utf-8-sig").splitlines() if s.strip()]
                    if not texts:
                        raise ValueError("Missing interaction captions")
                    arrays, betas, genders = [], [], []
                    stamps = []
                    for person in (role["actor_id"], role["reactor_id"]):
                        path = root / "motions" / episode / (person + ".npz")
                        stamps.append([path.stat().st_size, path.stat().st_mtime_ns])
                        with np.load(path, allow_pickle=False) as z:
                            gender = str(z["gender"].item()).lower()
                            genders.append(GENDERS.index(gender))
                            betas.append(z["betas"].reshape(-1)[:10].astype(np.float32))
                            stride = d["raw_fps"] // d["fps"]
                            trans = z["trans"][::stride].astype(np.float32)
                            poses = np.concatenate((z["root_orient"][::stride, None], z["pose_body"][::stride]), 1)
                            if not all(np.isfinite(x).all() for x in (trans, poses, betas[-1])):
                                raise ValueError("Nonfinite source motion")
                            rots = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
                            arrays.append((trans, rots.astype(np.float32)))
                    if len(arrays[0][0]) != len(arrays[1][0]):
                        raise ValueError("Actor/reactor frame count mismatch")
                    T = len(arrays[0][0])
                    if T < d["history"] + d["future"]:
                        raise ValueError("Sequence too short")
                    source_digest = digest({"stamps": stamps, "texts": texts, "role": role,
                                            "settings": settings})
                    valid_cache = False
                    if target.exists():
                        with np.load(target, allow_pickle=False) as saved:
                            valid_cache = str(saved["source_digest"]) == source_digest
                    if not valid_cache and reuse_settings is not None:
                        old_target = reuse_cache / "episodes" / (episode + ".npz")
                        old_digest = digest({"stamps": stamps, "texts": texts, "role": role,
                                             "settings": reuse_settings})
                        if old_target.exists():
                            with np.load(old_target, allow_pickle=False) as saved:
                                if str(saved["source_digest"]) == old_digest:
                                    payload = {key: saved[key] for key in saved.files}
                                    payload["source_digest"] = source_digest
                                    with target.with_suffix(".tmp").open("wb") as f:
                                        np.savez_compressed(f, **payload)
                                    target.with_suffix(".tmp").replace(target)
                                    valid_cache = True
                    if not valid_cache:
                        tr = torch.tensor(np.stack([x[0] for x in arrays]), device=device)
                        rr = torch.tensor(np.stack([x[1] for x in arrays]), device=device)
                        bb = torch.tensor(np.stack(betas), device=device)
                        gg = torch.tensor(genders, device=device)
                        offset = body.offsets(bb, gg)
                        joints = body.joints(tr, rr, bb, gg)
                        features = make_features(tr, rr, joints)
                        features = transform_features(features, torch.zeros(2, 3, device=device),
                                                      rotate_y_to_z.expand(2, -1, -1), offset, inverse=True)
                        embeddings = clip.encode(texts).numpy()
                        tmp = target.with_suffix(".tmp")
                        with tmp.open("wb") as f:
                            np.savez_compressed(f, features=features.cpu().numpy(), betas=np.stack(betas),
                                                genders=np.array(genders), offsets=offset.cpu().numpy(),
                                                captions=np.array(texts), text_embeddings=embeddings,
                                                source_digest=source_digest, fps=d["fps"])
                        tmp.replace(target)
                    manifest.append({"episode": episode, "split": split, "frames": T,
                                     "file": str(target.relative_to(cache)), "source_digest": source_digest, **role})
                except (ValueError, KeyError, FileNotFoundError) as error:
                    skipped.append({"episode": episode, "split": split, "reason": str(error)})
    manifest_digest = digest({"settings": settings, "split_digest": split_digest, "records": manifest})
    meta = {"settings": settings, "official_split_digest": split_digest,
            "digest": manifest_digest, "records": manifest, "skipped": skipped}
    dump_json(cache / "manifest.json", meta)
    compute_stats(cache, meta, d["history"], d["future"], stride=d.get("stride", 1))
    report = {"digest": manifest_digest, "feature_version": FEATURE_VERSION,
              "counts": {s: sum(r["split"] == s for r in manifest) for s in splits},
              "role_counts": {s: sum(r["role_status"] == s for r in manifest)
                              for s in ("annotation_assumed", "motion_verified")},
              "skipped": skipped, "normalization_split": "train", "limit_per_split": limit,
              "normalization_version": STATS_VERSION, "normalization_role_weights": [0.5, 0.5],
              "normalization_stride": d.get("stride", 1),
              "note": "Direction follows README plus verified overrides; unreviewed roles are assumptions."}
    dump_json(cache / "preparation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def compute_stats(cache, manifest, history, future, stride=1):
    """Equal actor/reactor frame counts in shared, history-defined coordinates."""
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("Statistics stride must be a positive integer")
    total = torch.zeros(276, dtype=torch.float64)
    square = torch.zeros_like(total)
    count = 0
    for record in manifest["records"]:
        if record["split"] != "train":
            continue
        with np.load(Path(cache) / record["file"]) as z:
            x = torch.from_numpy(z["features"])
            offsets = torch.from_numpy(z["offsets"])
        if x.shape[1] < history + future:
            continue
        starts = torch.arange(0, x.shape[1] - history - future + 1, stride)
        for chunk in starts.split(128):
            ids = chunk[:, None] + torch.arange(history + future)[None]
            a, r = x[0][ids], x[1][ids]
            origin, basis = reference_frame(r[:, :history])
            a = transform_features(a, origin, basis, offsets[0].expand(len(a), -1))
            r = transform_features(r, origin, basis, offsets[1].expand(len(r), -1))
            values = torch.cat((a, r), 1).reshape(-1, 276).double()
            total += values.sum(0)
            square += values.square().sum(0)
            count += len(values)
    if count == 0:
        raise ValueError("No train windows available for normalization")
    mean = total / count
    std = (square / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-3)
    target = Path(cache) / "stats.npz"
    with target.with_suffix(".tmp").open("wb") as f:
        np.savez(f, mean=mean.float().numpy(), std=std.float().numpy(),
                 count=count, split="train", manifest_digest=manifest["digest"],
                 version=STATS_VERSION, stride=stride, history=history, future=future,
                 role_weights=np.array([0.5, 0.5]), role_counts=np.array([count // 2, count // 2]))
    target.with_suffix(".tmp").replace(target)


class InterXDataset(Dataset):
    def __init__(self, cfg, split="train", primitives=None):
        self.cfg, self.split = cfg, split
        self.cache = Path(cfg["data"]["cache"])
        self.manifest = json.loads((self.cache / "manifest.json").read_text())
        d = cfg["data"]
        for key in ("fps", "history", "future"):
            if self.manifest["settings"][key] != d[key]:
                raise ValueError("Cache/config mismatch: " + key)
        if self.manifest["settings"]["feature_version"] != FEATURE_VERSION:
            raise ValueError("Incompatible motion features")
        self.length = d["history"] + (primitives or d["primitives"]) * d["future"]
        self.records = [r for r in self.manifest["records"] if r["split"] == split and r["frames"] >= self.length]
        if not self.records:
            raise ValueError("No sufficiently long episodes in split " + split)
        self.windows = [(i, start) for i, r in enumerate(self.records)
                        for start in range(0, r["frames"] - self.length + 1, d.get("stride", 1))]

    @lru_cache(maxsize=64)
    def episode(self, index):
        with np.load(self.cache / self.records[index]["file"], allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        ep, start = self.windows[index]
        z = self.episode(ep)
        caption = np.random.randint(len(z["captions"])) if self.split == "train" else 0
        return {"actor": torch.from_numpy(z["features"][0, start:start + self.length].copy()),
                "reactor": torch.from_numpy(z["features"][1, start:start + self.length].copy()),
                "betas": torch.from_numpy(z["betas"].copy()),
                "genders": torch.from_numpy(z["genders"].copy()),
                "offsets": torch.from_numpy(z["offsets"].copy()),
                "text": torch.from_numpy(z["text_embeddings"][caption].copy())}


def load_stats(cache, device):
    cache = Path(cache)
    manifest = json.loads((cache / "manifest.json").read_text())
    with np.load(cache / "stats.npz") as z:
        if str(z["manifest_digest"]) != manifest["digest"] or str(z["split"]) != "train":
            raise ValueError("Normalization provenance mismatch")
        return {k: torch.tensor(z[k], device=device) for k in ("mean", "std")}


def condition_window(actor, reactor, offsets, stats):
    origin, basis = reference_frame(reactor)
    a = transform_features(actor, origin, basis, offsets[:, 0])
    r = transform_features(reactor, origin, basis, offsets[:, 1])
    return ((a - stats["mean"]) / stats["std"],
            (r - stats["mean"]) / stats["std"], origin, basis)
