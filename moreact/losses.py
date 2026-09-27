"""Reactor reconstruction losses for the conditional motion VAE."""
from __future__ import annotations

import torch
from contextvars import ContextVar
from torch.nn import functional as F

from .geometry import (DJOINTS, DROT, DTRANS, JOINTS, POSE, TRANSL,
                       matrix_to_6d, rotation_6d_to_matrix)


# SMPL-X body-only joint tree used by the 22-joint motion representation.
BODY_BONES = tuple((child, parent) for child, parent in enumerate(
    (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)) if parent >= 0)
FOOT_JOINTS = (7, 8, 10, 11)
FOOT_HEIGHTS = (0.12, 0.12, 0.05, 0.05)
GLOBAL_MASK_NORMALIZATION = ContextVar('global_mask_normalization', default=False)


def huber(prediction, target):
    return F.huber_loss(prediction, target, delta=1.0)


def feature_reconstruction_loss(prediction, target, root_weight=1.):
    """Weight 24 root channels; keep the original B*T*276 denominator."""
    if root_weight == 1.:
        return huber(prediction, target)
    weights = prediction.new_ones(prediction.shape[-1])
    for channels in (TRANSL, slice(POSE.start, POSE.start + 6), DTRANS, DROT,
                     slice(JOINTS.start, JOINTS.start + 3),
                     slice(DJOINTS.start, DJOINTS.start + 3)):
        weights[channels] = root_weight
    return (F.huber_loss(prediction, target, delta=1., reduction='none') * weights).mean()


def masked_huber(prediction, target):
    """Preserve the global contact-element mean under DDP gradient averaging."""
    numerator = F.huber_loss(prediction, target, delta=1., reduction='sum')
    count = prediction.new_tensor(prediction.numel())
    world = 1
    if GLOBAL_MASK_NORMALIZATION.get():
        import torch.distributed as dist
        dist.all_reduce(count)
        world = dist.get_world_size()
    return numerator * world / count.clamp_min(1)


def bone_length_loss(pred_joints, target_joints):
    children, parents = zip(*BODY_BONES)
    pred = (pred_joints[..., children, :] - pred_joints[..., parents, :]).norm(dim=-1)
    target = (target_joints[..., children, :] - target_joints[..., parents, :]).norm(dim=-1)
    return huber(pred, target)


def foot_contact_loss(pred_joints, target_joints, previous_target_joints):
    """Penalize generated foot displacement where the target foot is in contact.

    Coordinates are Z-up. Contact follows the InterGen thresholds, but is detected
    from the target rather than the prediction so the model cannot evade the loss.
    The last observed history frame is included in the first future displacement.
    """
    pred = torch.cat((previous_target_joints[:, None], pred_joints), dim=1)
    target = torch.cat((previous_target_joints[:, None], target_joints), dim=1)
    pred_velocity = pred[:, 1:, FOOT_JOINTS] - pred[:, :-1, FOOT_JOINTS]
    target_velocity = target[:, 1:, FOOT_JOINTS] - target[:, :-1, FOOT_JOINTS]
    heights = target[:, 1:, FOOT_JOINTS, 2]
    height_limit = target.new_tensor(FOOT_HEIGHTS)
    contact = (target_velocity.square().sum(dim=-1) < 0.001) & (heights < height_limit)
    return masked_huber(pred_velocity[contact], torch.zeros_like(pred_velocity[contact]))


def vae_reconstruction_losses(prediction, target, history, mean, std, body,
                              betas, genders, weights):
    """Compute unweighted CVAE terms and their configured weighted total.

    All inputs use [B,T,D]. Prediction, target, and history are normalized in the
    shared reactor frame. Deltas use MoReAct's causal convention: the value stored
    at frame t describes frame t minus frame t-1.
    """
    terms = {"feature_rec": huber(prediction, target)}
    pred = prediction * std + mean
    truth = target * std + mean
    previous = history[:, -1] * std + mean

    pred_joints = pred[..., JOINTS].reshape(*pred.shape[:2], 22, 3)
    target_joints = truth[..., JOINTS].reshape(*truth.shape[:2], 22, 3)
    previous_joints = previous[..., JOINTS].reshape(len(previous), 22, 3)

    rotations = rotation_6d_to_matrix(pred[..., POSE].reshape(*pred.shape[:2], 22, 6))
    fk_joints = body.joints(pred[..., TRANSL], rotations, betas, genders)
    terms["smpl_joints_rec"] = huber(fk_joints, target_joints)
    terms["joint_fk_consistency"] = huber(pred_joints, fk_joints)

    sequence_transl = torch.cat((previous[:, None, TRANSL], pred[..., TRANSL]), dim=1)
    calculated_transl_delta = sequence_transl[:, 1:] - sequence_transl[:, :-1]
    terms["transl_delta"] = huber(pred[..., DTRANS], calculated_transl_delta)

    sequence_joints = torch.cat((previous_joints[:, None], pred_joints), dim=1)
    calculated_joints_delta = sequence_joints[:, 1:] - sequence_joints[:, :-1]
    terms["joints_delta"] = huber(
        pred[..., DJOINTS].reshape(*pred.shape[:2], 22, 3), calculated_joints_delta)

    previous_root = rotation_6d_to_matrix(previous[..., POSE].reshape(len(previous), 22, 6))[:, 0]
    root = rotations[..., 0, :, :]
    root_sequence = torch.cat((previous_root[:, None], root), dim=1)
    calculated_orient_delta = root_sequence[:, 1:] @ root_sequence[:, :-1].transpose(-1, -2)
    terms["orient_delta"] = huber(pred[..., DROT], matrix_to_6d(calculated_orient_delta))

    terms["bone_length"] = bone_length_loss(pred_joints, target_joints)
    terms["foot_contact"] = foot_contact_loss(pred_joints, target_joints, previous_joints)
    total = sum(weights[name] * value for name, value in terms.items())
    return total, terms


def diffusion_motion_losses(prediction, target, history, mean, std, body,
                            betas, genders, weights, actor_future=None, options=None,
                            target_history=None):
    """Decoded reactor supervision; temporal differences include the boundary.

    Geometry is measured in metres, velocity in metres/frame (not metres/sec).
    Frozen decoder/FK parameters must still allow derivatives wrt prediction.
    """
    options = options or {}
    vae_keys = ("feature_rec", "smpl_joints_rec", "joint_fk_consistency",
                "transl_delta", "joints_delta", "orient_delta", "bone_length", "foot_contact")
    _, terms = vae_reconstruction_losses(prediction, target, history, mean, std,
                                         body, betas, genders, {k: 0. for k in vae_keys})
    terms['feature_rec'] = feature_reconstruction_loss(
        prediction, target, options.get('feature_root_weight', 1.))
    previous = (history[:, -1:] * std + mean)[..., JOINTS]
    pred = torch.cat((previous, (prediction * std + mean)[..., JOINTS]), 1)
    truth = torch.cat((previous, (target * std + mean)[..., JOINTS]), 1)
    terms['joint_velocity'] = huber(pred[:, 1:] - pred[:, :-1], truth[:, 1:] - truth[:, :-1])
    physical = prediction * std + mean
    target_physical = target * std + mean
    # Same shaped pelvis offset and rigid frame on both sides: translation
    # error equals pelvis-position error, with rotation-invariant squared norm.
    if weights.get('root_position', 0) > 0:
        terms['root_position'] = F.mse_loss(
            physical[..., TRANSL], target_physical[..., TRANSL])
    pred_rot = rotation_6d_to_matrix(physical[..., POSE].reshape(*prediction.shape[:2], 22, 6))
    gt_rot = rotation_6d_to_matrix(target_physical[..., POSE].reshape(*target.shape[:2], 22, 6))
    need_fk = any(weights.get(key, 0) > 0 for key in
                  ('root_relative_translation', 'distance_map', 'joint_contact'))
    fk = body.joints(physical[..., TRANSL], pred_rot, betas, genders) if need_fk else None
    # Chordal rotation error: matrix-space MSE, without an unstable acos.
    terms['root_orientation'] = F.mse_loss(pred_rot[:, :, 0], gt_rot[:, :, 0])
    prev = rotation_6d_to_matrix((history[:, -1:] * std + mean)[..., POSE].reshape(len(history), 1, 22, 6))[:, :, 0]
    rp = torch.cat((prev, pred_rot[:, :, 0]), 1)
    rt = torch.cat((prev, gt_rot[:, :, 0]), 1)
    terms['root_angular_velocity'] = F.mse_loss(rp[:, 1:] @ rp[:, :-1].transpose(-1,-2),
                                               rt[:, 1:] @ rt[:, :-1].transpose(-1,-2))
    if weights.get('root_relative_translation', 0) > 0 or weights.get('root_relative_orientation', 0) > 0:
        if target_history is None:
            raise ValueError('Relative root losses require original GT history')
        true_previous = target_history[:, -1] * std + mean
        horizons = [h - 1 for h in (1, 4, 8) if h <= prediction.shape[1]]
        if weights.get('root_relative_translation', 0) > 0:
            # Each side uses its own history anchor. An inherited rigid offset
            # therefore does not become a one-frame catch-up target.
            pred_root = fk[..., 0, :]
            prior_pred = previous[:, 0].reshape(len(history), 22, 3)[:, 0]
            prior_true = true_previous[..., JOINTS.start:JOINTS.start + 3]
            true_root = target_physical[..., JOINTS.start:JOINTS.start + 3]
            terms['root_relative_translation'] = huber(
                pred_root[:, horizons] - prior_pred[:, None],
                true_root[:, horizons] - prior_true[:, None])
        if weights.get('root_relative_orientation', 0) > 0:
            true_previous_root = rotation_6d_to_matrix(
                true_previous[..., POSE.start:POSE.start + 6])
            terms['root_relative_orientation'] = F.mse_loss(
                pred_rot[:, horizons, 0] @ prev[:, 0, None].transpose(-1, -2),
                gt_rot[:, horizons, 0] @ true_previous_root[:, None].transpose(-1, -2))
    if weights.get('distance_map', 0) > 0 or weights.get('joint_contact', 0) > 0:
        if actor_future is None:
            raise ValueError('Interaction losses require actor future labels in the shared frame')
        actor = actor_future.detach()[..., JOINTS].reshape(*target.shape[:2], 22, 3)
        gt_joints = target_physical.detach()[..., JOINTS].reshape(*target.shape[:2], 22, 3)
        dp = torch.cdist(actor, fk, compute_mode='donot_use_mm_for_euclid_dist')
        dt = torch.cdist(actor, gt_joints, compute_mode='donot_use_mm_for_euclid_dist')
        threshold = options.get('distance_map_threshold_m')
        if threshold is None:
            terms['distance_map'] = huber(dp, dt)
        else:
            # GT mask prevents escaping supervision by predicting large distances.
            nearby = dt < threshold
            terms['distance_map'] = masked_huber(dp[nearby], dt[nearby])
        # GT joint proximity is a contact proxy, not mesh contact/penetration.
        contact = dt < 0.10
        terms['joint_contact'] = masked_huber(dp[contact], dt[contact])
    terms = {name: value for name, value in terms.items() if weights.get(name, 0) > 0}
    return sum(weights[name] * value for name, value in terms.items()), terms
