"""Training-only corruption of physical reactor history features."""
from __future__ import annotations

import torch

from .geometry import (DJOINTS, DROT, DTRANS, JOINTS, POSE, TRANSL,
                       matrix_to_6d, rotation_6d_to_matrix)


def translate_history(history, probability=0., std_m=0.02, max_m=0.05):
    """Apply one bounded XY shift per sample, shared by all history frames.

    Translation and joints move together; rotations and internal deltas are
    unchanged. The first stored delta also assumes the predecessor was shifted.
    Inputs are world-frame, unnormalized features. Never mutate the caller.
    """
    if probability == 0 or std_m == 0:
        return history, torch.zeros(len(history), dtype=torch.bool, device=history.device)
    selected = torch.rand(len(history), device=history.device) < probability
    shift = torch.randn(len(history), 2, device=history.device, dtype=history.dtype) * std_m
    shift = shift * (max_m / shift.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.)
    shift = torch.cat((shift, torch.zeros_like(shift[:, :1])), -1) * selected[:, None]
    result = history.clone()
    result[..., TRANSL] += shift[:, None]
    joints = result[..., JOINTS].reshape(len(history), history.shape[1], 22, 3)
    result[..., JOINTS] = (joints + shift[:, None, None]).flatten(-2)
    return result, selected


def repair_target_boundary(target, history, selected):
    """Make selected GT futures' first causal deltas match the supplied history.

    Positions/poses stay at the original GT. This is small-offset recovery
    supervision, not a synthesized smooth recovery trajectory.
    """
    result = target.clone()
    first, previous = target[:, 0], history[:, -1]
    for position, delta in ((TRANSL, DTRANS), (JOINTS, DJOINTS)):
        value = first[..., position] - previous[..., position]
        result[:, 0, delta] = torch.where(selected[:, None], value, first[..., delta])
    root = rotation_6d_to_matrix(first[..., POSE.start:POSE.start + 6])
    previous_root = rotation_6d_to_matrix(previous[..., POSE.start:POSE.start + 6])
    delta = matrix_to_6d(root @ previous_root.transpose(-1, -2))
    result[:, 0, DROT] = torch.where(selected[:, None], delta, first[..., DROT])
    return result
