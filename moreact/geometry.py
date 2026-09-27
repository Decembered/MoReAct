"""DART feature layout with causal deltas and a shared, Z-up pair frame."""
from __future__ import annotations

import torch
from torch.nn import functional as F

DIM = 276
TRANSL = slice(0, 3)
POSE = slice(3, 135)
DTRANS = slice(135, 138)
DROT = slice(138, 144)
JOINTS = slice(144, 210)
DJOINTS = slice(210, 276)
GENDERS = ("male", "female", "neutral")


def matrix_to_6d(matrix):
    # Same row convention as PyTorch3D / DART.
    return matrix[..., :2, :].clone().flatten(-2)


def rotation_6d_to_matrix(d6):
    a, b = d6[..., :3], d6[..., 3:]
    # Deterministic finite fallback for an untrained model's degenerate outputs.
    e1 = torch.zeros_like(a)
    e1[..., 0] = 1
    a = torch.where(a.norm(dim=-1, keepdim=True) > 1e-6, a, e1)
    a = F.normalize(a, dim=-1)
    b = b - (a * b).sum(-1, keepdim=True) * a
    fallback = F.one_hot(a.abs().argmin(-1), 3).to(a.dtype)
    fallback = fallback - (a * fallback).sum(-1, keepdim=True) * a
    b = torch.where(b.norm(dim=-1, keepdim=True) > 1e-6, b, fallback)
    b = F.normalize(b, dim=-1)
    return torch.stack((a, b, torch.cross(a, b, dim=-1)), -2)


def make_features(trans, rotations, joints):
    """Accept [..., T, 3], [..., T, 22, 3, 3], [..., T, 22, 3]."""
    dt = torch.zeros_like(trans)
    dj = torch.zeros_like(joints)
    dt[..., 1:, :] = trans[..., 1:, :] - trans[..., :-1, :]
    dj[..., 1:, :, :] = joints[..., 1:, :, :] - joints[..., :-1, :, :]
    root = rotations[..., 0, :, :]
    dr = torch.eye(3, device=trans.device, dtype=trans.dtype).expand_as(root).clone()
    dr[..., 1:, :, :] = root[..., 1:, :, :] @ root[..., :-1, :, :].transpose(-1, -2)
    return torch.cat((trans, matrix_to_6d(rotations).flatten(-2), dt,
                      matrix_to_6d(dr), joints.flatten(-2), dj.flatten(-2)), -1)


def reference_frame(history):
    """Frame from last observed pelvis/hips; origin keeps the original floor."""
    joints = history[:, -1, JOINTS].reshape(-1, 22, 3)
    origin = joints[:, 0].clone()
    origin[:, 2] = 0
    x = joints[:, 2] - joints[:, 1]
    x[:, 2] = 0
    # Root orientation provides an observed fallback if hips coincide.
    root = rotation_6d_to_matrix(history[:, -1, POSE].reshape(-1, 22, 6))[:, 0]
    fallback = root[:, :, 0].clone()
    fallback[:, 2] = 0
    unit = torch.zeros_like(x)
    unit[:, 0] = 1
    fallback = torch.where(fallback.norm(dim=-1, keepdim=True) > 1e-6, fallback, unit)
    x = torch.where(x.norm(dim=-1, keepdim=True) > 1e-6, x, fallback)
    x = F.normalize(x, dim=-1)
    z = torch.zeros_like(x)
    z[:, 2] = 1
    y = torch.cross(z, x, dim=-1)
    return origin, torch.stack((x, y, z), -1)


def transform_features(features, origin, basis, pelvis_offset, inverse=False):
    """Batched rigid transform. SMPL transl rotates about its shaped pelvis."""
    out = features.clone()
    B, T = features.shape[:2]
    rot = basis if inverse else basis.transpose(-1, -2)
    shift = origin[:, None, :]
    offset = pelvis_offset[:, None, :]

    def vectors(v):
        return torch.einsum("bij,btj->bti", rot, v)

    if inverse:
        out[..., TRANSL] = vectors(features[..., TRANSL] + offset) + shift - offset
    else:
        out[..., TRANSL] = vectors(features[..., TRANSL] + offset - shift) - offset
    joints = features[..., JOINTS].reshape(B, T * 22, 3)
    joints = vectors(joints) + shift if inverse else vectors(joints - shift)
    out[..., JOINTS] = joints.reshape(B, T, 66)
    out[..., DTRANS] = vectors(features[..., DTRANS])
    out[..., DJOINTS] = vectors(features[..., DJOINTS].reshape(B, T * 22, 3)).reshape(B, T, 66)
    poses = rotation_6d_to_matrix(features[..., POSE].reshape(B, T, 22, 6))
    poses = poses.clone()
    poses[:, :, 0] = rot[:, None] @ poses[:, :, 0]
    out[..., POSE] = matrix_to_6d(poses).flatten(-2)
    delta = rotation_6d_to_matrix(features[..., DROT])
    out[..., DROT] = matrix_to_6d(rot[:, None] @ delta @ rot[:, None].transpose(-1, -2))
    return out


class BodyModels:
    """Lazy SMPL-X FK; no asset paths or body models imported globally."""

    def __init__(self, root, device="cpu"):
        self.root, self.device, self.models = str(root), torch.device(device), {}

    def model(self, gender):
        if gender not in self.models:
            import smplx
            model = smplx.build_layer(self.root, model_type="smplx", gender=gender,
                                     ext="npz", use_pca=False, flat_hand_mean=True)
            self.models[gender] = model.to(self.device).eval().requires_grad_(False)
        return self.models[gender]

    def joints(self, trans, rotations, betas, genders):
        """Run differentiable SMPL-X FK with frozen body-model parameters."""
        B, T = trans.shape[:2]
        result = torch.empty(B, T, 22, 3, device=trans.device, dtype=trans.dtype)
        for gender_id, gender in enumerate(GENDERS):
            ids = (genders == gender_id).nonzero(as_tuple=True)[0]
            if not len(ids):
                continue
            tr = trans[ids].reshape(-1, 3)
            rr = rotations[ids].reshape(-1, 22, 3, 3)
            bb = betas[ids, None].expand(-1, T, -1).reshape(-1, 10)
            chunks = []
            model = self.model(gender)
            for start in range(0, len(tr), 128):
                end = start + 128
                output = model(transl=tr[start:end], betas=bb[start:end],
                               global_orient=rr[start:end, :1], body_pose=rr[start:end, 1:],
                               return_verts=False)
                chunks.append(output.joints[:, :22])
            result[ids] = torch.cat(chunks).reshape(len(ids), T, 22, 3)
        return result

    @torch.no_grad()
    def offsets(self, betas, genders):
        B = len(betas)
        rotations = torch.eye(3, device=betas.device).expand(B, 1, 22, 3, 3)
        return self.joints(torch.zeros(B, 1, 3, device=betas.device), rotations,
                           betas, genders)[:, 0, 0]

    @torch.no_grad()
    def repair(self, generated, previous, betas, genders):
        """Project poses, recompute FK and causal deltas before feedback."""
        joined = torch.cat((previous[:, -1:], generated), 1)
        rotations = rotation_6d_to_matrix(joined[..., POSE].reshape(*joined.shape[:2], 22, 6))
        joints = self.joints(joined[..., TRANSL], rotations, betas, genders)
        return make_features(joined[..., TRANSL], rotations, joints)[:, 1:]
