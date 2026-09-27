from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..config import digest, dump_json

VERSION = "shared_cvae_mean_rvq_v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def new_output(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=False)
    return p


def read_manifest(root, kind=None):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    if m.get("version") != VERSION or (kind is not None and m.get("kind") != kind):
        raise ValueError("Incompatible semantic cache")
    if m.get("manifest_id") != digest({k: v for k, v in m.items() if k != "manifest_id"}):
        raise ValueError("Semantic manifest identity mismatch")
    seen = set()
    for r in m["records"]:
        if r["episode"] in seen or r["split"] not in ("train", "val", "test"):
            raise ValueError("Duplicate episode or invalid split")
        seen.add(r["episode"])
        file = (root / r["file"]).resolve()
        if root.resolve() not in file.parents or not file.is_file() or sha256(file) != r["sha256"]:
            raise ValueError("Cache file identity mismatch: " + r["file"])
    return m


def write_manifest(root, manifest):
    manifest = dict(manifest, version=VERSION)
    manifest["manifest_id"] = digest(manifest)
    dump_json(Path(root) / "manifest.json", manifest)
    return manifest


def load_arrays(root, record):
    with np.load(Path(root) / record["file"], allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def require_identity(expected, actual, name="CVAE"):
    if expected != actual:
        raise ValueError(name + " identity mismatch")
