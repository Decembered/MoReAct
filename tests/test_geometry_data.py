from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from moreact.data import (InterXDataset, compute_stats, condition_window, load_stats,
                         read_splits, resolve_role)
from moreact.geometry import (DROT, DTRANS, JOINTS, make_features, matrix_to_6d,
                              reference_frame, rotation_6d_to_matrix, transform_features)


def test_causal_differences():
    trans = torch.randn(1, 12, 3)
    rot = rotation_6d_to_matrix(torch.randn(1, 12, 22, 6))
    joints = torch.randn(1, 12, 22, 3)
    original = make_features(trans, rot, joints)
    trans[:, 6:] += 100
    rot[:, 6:] = torch.eye(3)
    joints[:, 6:] -= 50
    changed = make_features(trans, rot, joints)
    torch.testing.assert_close(original[:, :6], changed[:, :6], rtol=0, atol=0)
    torch.testing.assert_close(original[:, 0, DTRANS], torch.zeros(1, 3))
    torch.testing.assert_close(original[:, 0, DROT], matrix_to_6d(torch.eye(3))[None])


def test_rigid_round_trip_and_relative_geometry():
    torch.manual_seed(4)
    trans = torch.randn(2, 7, 3)
    rot = rotation_6d_to_matrix(torch.randn(2, 7, 22, 6))
    joints = torch.randn(2, 7, 22, 3)
    x = make_features(trans, rot, joints)
    origin = torch.randn(1, 3).expand(2, -1)
    basis = rotation_6d_to_matrix(torch.randn(1, 6)).expand(2, -1, -1)
    offsets = torch.randn(2, 3)
    local = transform_features(x, origin, basis, offsets)
    restored = transform_features(local, origin, basis, offsets, inverse=True)
    torch.testing.assert_close(restored, x, rtol=1e-4, atol=3e-5)
    before = (joints[0] - joints[1]).norm(dim=-1)
    after = (local[0, :, JOINTS].reshape(-1, 22, 3) - local[1, :, JOINTS].reshape(-1, 22, 3)).norm(dim=-1)
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)


def test_degenerate_six_d_is_valid_rotation():
    rotation = rotation_6d_to_matrix(torch.zeros(7, 6))
    torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), torch.eye(3).expand(7, -1, -1))
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(7))


def test_roles_verified_override_and_unverified_rejection():
    assert resolve_role("x", 0, {})["actor_id"] == "P1"
    assert resolve_role("x", 1, {})["actor_id"] == "P2"
    overrides = {"x": {"actor": "P1", "status": "motion_verified", "evidence": "raw motion review"}}
    assert resolve_role("x", 1, overrides)["reactor_id"] == "P2"
    overrides["x"]["status"] = "candidate"
    with pytest.raises(ValueError):
        resolve_role("x", 1, overrides)
    actual = json.loads((Path(__file__).resolve().parents[1] / "assets/role_overrides.json").read_text())
    assert resolve_role("G056T001A005R001", 1, actual)["actor_id"] == "P1"


def test_split_leakage_rejected(tmp_path):
    (tmp_path / "splits").mkdir()
    for split, names in (("train", "a\nb\n"), ("val", "b\n"), ("test", "c\n")):
        (tmp_path / "splits" / (split + ".txt")).write_text(names)
    with pytest.raises(ValueError, match="leakage"):
        read_splits(tmp_path)


def test_normalization_ignores_val_test(tiny_config):
    cache = Path(tiny_config["data"]["cache"])
    before = load_stats(cache, "cpu")
    meta = json.loads((cache / "manifest.json").read_text())
    for record in meta["records"]:
        if record["split"] != "train":
            with np.load(cache / record["file"]) as z:
                data = {k: z[k] for k in z.files}
            data["features"][:] = 1e6
            np.savez(cache / record["file"], **data)
    compute_stats(cache, meta, 4, 2)
    after = load_stats(cache, "cpu")
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_conditions_invariant_to_common_yaw_translation(tiny_config):
    data = InterXDataset(tiny_config)[0]
    a, r = data["actor"][:4][None], data["reactor"][:4][None]
    stats = load_stats(tiny_config["data"]["cache"], "cpu")
    offsets = data["offsets"][None]
    initial = condition_window(a, r, offsets, stats)
    basis = torch.tensor([[[0., -1, 0], [1, 0, 0], [0, 0, 1]]])
    shift = torch.tensor([[3., -2, 0]])
    a2 = transform_features(a, shift, basis, offsets[:, 0], inverse=True)
    r2 = transform_features(r, shift, basis, offsets[:, 1], inverse=True)
    updated = condition_window(a2, r2, offsets, stats)
    for x, y in zip(initial[:2], updated[:2]):
        torch.testing.assert_close(x, y, rtol=1e-3, atol=1e-3)


def test_equal_role_statistics_match_explicit_population(tiny_config):
    cache = Path(tiny_config["data"]["cache"])
    meta = json.loads((cache / "manifest.json").read_text())
    record = next(r for r in meta["records"] if r["split"] == "train")
    with np.load(cache / record["file"]) as z:
        features = torch.from_numpy(z["features"])
        offsets = torch.from_numpy(z["offsets"])
    values = []
    for start in range(features.shape[1] - 6 + 1):
        origin, basis = reference_frame(features[1, start:start + 4][None])
        for role in (0, 1):
            local = transform_features(features[role, start:start + 6][None],
                                       origin, basis, offsets[role][None])
            values.append(local.reshape(-1, 276).double())
    population = torch.cat(values)
    with np.load(cache / "stats.npz") as z:
        torch.testing.assert_close(torch.from_numpy(z["mean"]), population.mean(0).float())
        torch.testing.assert_close(torch.from_numpy(z["std"]),
                                   population.std(0, unbiased=False).clamp_min(1e-3).float())
        assert z["role_counts"].tolist() == [66, 66]
        assert z["role_weights"].tolist() == [0.5, 0.5]
        assert int(z["stride"]) == 1


def test_sampling_stride_one(tiny_config):
    dataset = InterXDataset(tiny_config)
    assert dataset.windows == [(0, start) for start in range(9)]
