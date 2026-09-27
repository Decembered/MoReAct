import numpy as np
import pytest

from moreact.evaluate import motion_metrics
from moreact.geometry import JOINTS


def test_constant_velocity_has_no_boundary_jump_or_reference_error():
    motion = np.zeros((12, 276), dtype=np.float32)
    joints = np.zeros((12, 22, 3), dtype=np.float32)
    joints[:, :, 0] = np.arange(1, 13)[:, None] / 30
    motion[:, JOINTS] = joints.reshape(12, 66)
    metrics = motion_metrics(motion, motion, motion, 30, 4, np.zeros(276))
    assert metrics["world_mpjpe_m"] == 0
    assert metrics["root_ade_m"] == 0
    assert metrics["boundary_velocity_jump_mps"] < 1e-5
    assert metrics["foot_sliding_mps"] == pytest.approx(1., abs=1e-5)


def test_no_predicted_floor_contact_reports_missing_not_zero_sliding():
    motion = np.zeros((8, 276), dtype=np.float32)
    joints = np.zeros((8, 22, 3), dtype=np.float32)
    joints[:, :, 2] = 1.
    motion[:, JOINTS] = joints.reshape(8, 66)
    metrics = motion_metrics(motion, motion, motion, 30, 4)
    assert metrics["foot_sliding_mps"] is None
    assert metrics["foot_contact_fraction"] == 0
