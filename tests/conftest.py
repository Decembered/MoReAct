from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moreact.config import load_config, dump_json
from moreact.data import FEATURE_VERSION, compute_stats
from moreact.geometry import make_features


@pytest.fixture
def tiny_config(tmp_path):
    cfg = load_config()
    cfg["data"].update(cache=str(tmp_path / "data"), history=4, future=2, primitives=2)
    cfg["model"].update(vae_hidden=32, vae_layers=3, denoiser_hidden=32,
                         denoiser_layers=2, ff_size=64, latent_dim=8, dropout=0., diffusion_steps=3)
    cfg["train"].update(batch_size=2, stage1_steps=10, stage2_steps=0, stage3_steps=0,
                         val_batches=1, latent_scale_batches=2, val_interval=2,
                         log_interval=2, save_interval=2, device="cpu", threads=1)
    cache = Path(cfg["data"]["cache"])
    (cache / "episodes").mkdir(parents=True)
    records = []
    torch.manual_seed(321)
    for i, split in enumerate(("train", "val", "test")):
        T = 16
        trans = torch.randn(2, T, 3).cumsum(1) * .01
        trans[..., 2] += 1.
        trans[0, :, 0] += 1.
        rotation = torch.eye(3).expand(2, T, 22, 3, 3)
        shape = torch.randn(22, 3) * .1
        shape[0] = 0
        shape[1] = torch.tensor([-.1, 0, 0])
        shape[2] = torch.tensor([.1, 0, 0])
        joints = trans[:, :, None] + shape
        features = make_features(trans, rotation, joints)
        file = "episodes/sample%d.npz" % i
        np.savez(cache / file, features=features.numpy(), betas=np.zeros((2, 10), np.float32),
                 offsets=np.zeros((2, 3), np.float32), genders=np.array([0, 1]),
                 captions=np.array(["A approaches B"]), text_embeddings=np.ones((1, 512), np.float32))
        records.append({"episode": "sample%d" % i, "split": split, "frames": T, "file": file})
    meta = {"settings": {"history": 4, "future": 2, "fps": 30, "feature_version": FEATURE_VERSION},
            "digest": "synthetic-only", "records": records}
    dump_json(cache / "manifest.json", meta)
    compute_stats(cache, meta, 4, 2)
    return cfg
