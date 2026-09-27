from __future__ import annotations

import torch

from moreact.geometry import DJOINTS, JOINTS, make_features
from moreact.losses import vae_reconstruction_losses


class FixedBody:
    def __init__(self, joints):
        self.fixed_joints = joints

    def joints(self, trans, rotations, betas, genders):
        return self.fixed_joints + trans[..., None, :] * 0.0


def test_cvae_physical_losses_include_history_boundary_and_gt_foot_contact():
    batch, history_length, future_length = 1, 2, 3
    frames = history_length + future_length
    trans = torch.zeros(batch, frames, 3)
    rotations = torch.eye(3).expand(batch, frames, 22, 3, 3).clone()
    shape = torch.zeros(22, 3)
    shape[:, 2] = 0.5
    shape[[7, 8, 10, 11], 2] = 0.0
    shape[:, 0] = torch.arange(22) * 0.01
    joints = trans[..., None, :] + shape
    features = make_features(trans, rotations, joints)
    history, target = features[:, :history_length], features[:, history_length:]
    target_joints = target[..., JOINTS].reshape(batch, future_length, 22, 3)
    weights = {name: 1.0 for name in ("feature_rec", "smpl_joints_rec",
              "joint_fk_consistency", "transl_delta", "joints_delta",
              "orient_delta", "bone_length", "foot_contact")}
    args = (target, history, torch.zeros(276), torch.ones(276),
            FixedBody(target_joints), torch.zeros(batch, 10),
            torch.zeros(batch, dtype=torch.long), weights)

    total, terms = vae_reconstruction_losses(target.clone(), *args)
    assert total == 0
    assert all(value == 0 for value in terms.values())

    inconsistent_delta = target.clone()
    inconsistent_delta[:, 0, DJOINTS] += 0.1
    _, delta_terms = vae_reconstruction_losses(inconsistent_delta, *args)
    assert delta_terms["joints_delta"] > 0

    moving_foot = target.clone()
    moving_foot[:, 0, JOINTS.start + 7 * 3] += 0.1
    _, foot_terms = vae_reconstruction_losses(moving_foot, *args)
    assert foot_terms["bone_length"] > 0
    assert foot_terms["foot_contact"] > 0


def test_interaction_labels_only_and_contact_distance_matching():
    from moreact.losses import diffusion_motion_losses
    trans = torch.zeros(1, 4, 3)
    rotations = torch.eye(3).expand(1,4,22,3,3)
    joints = torch.zeros(1,4,22,3)
    features = make_features(trans,rotations,joints)
    history,target = features[:,:2],features[:,2:]
    actor = target.clone()
    actor[..., JOINTS] = .04
    actor.requires_grad_()
    class TranslationBody:
        def joints(self, trans, rotations, betas, genders):
            return trans[:,:,None].expand(-1,-1,22,-1)
    weights={'distance_map':1.,'joint_contact':10.}
    def evaluate(pred):
        return diffusion_motion_losses(pred,target,history,torch.zeros(276),torch.ones(276),
            TranslationBody(),torch.zeros(1,10),torch.zeros(1,dtype=torch.long),weights,actor)
    loss,terms=evaluate(target)
    assert loss == 0 and all(v==0 for v in terms.values())
    pred=target.clone();pred[...,0] = .2;pred.requires_grad_()
    loss,terms=evaluate(pred)
    assert terms['joint_contact']>0 and terms['distance_map']>0
    loss.backward()
    assert pred.grad.abs().sum()>0 and actor.grad is None


def test_root_position_physical_scale_gradient_and_opt_out():
    from moreact.losses import diffusion_motion_losses
    trans = torch.zeros(1, 4, 3)
    rotations = torch.eye(3).expand(1, 4, 22, 3, 3)
    features = make_features(trans, rotations, torch.zeros(1, 4, 22, 3))
    mean, std = torch.zeros(276), torch.ones(276)
    mean[:3] = torch.tensor([1., 2., 3.])
    std[:3] = torch.tensor([2., 3., 4.])
    normalized = (features - mean) / std
    history, target = normalized[:, :2], normalized[:, 2:]
    def evaluate(pred, weights):
        return diffusion_motion_losses(pred, target, history, mean, std,
            FixedBody(torch.zeros(1, 2, 22, 3)), torch.zeros(1, 10),
            torch.zeros(1, dtype=torch.long), weights)
    loss, _ = evaluate(target, {'root_position': 30.})
    assert loss == 0
    pred = target.clone()
    pred[..., 0] += .15  # 0.30 metres after denormalization.
    pred.requires_grad_()
    loss, terms = evaluate(pred, {'root_position': 30.})
    torch.testing.assert_close(terms['root_position'], torch.tensor(.03))
    torch.testing.assert_close(loss, torch.tensor(.9))
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[..., 0].min() > 0
    assert pred.grad[..., 3:].count_nonzero() == 0
    for weights in ({'feature_rec': 1.}, {'feature_rec': 1., 'root_position': 0.}):
        _, terms = evaluate(pred.detach(), weights)
        assert set(terms) == {'feature_rec'}


def test_root_feature_weight_scales_only_root_channel_gradients():
    from moreact.losses import feature_reconstruction_loss
    prediction = torch.full((2, 3, 276), .2, requires_grad=True)
    target = torch.zeros_like(prediction)
    baseline = feature_reconstruction_loss(prediction, target)
    base_grad, = torch.autograd.grad(baseline, prediction)
    weighted = feature_reconstruction_loss(prediction, target, root_weight=5.)
    gradient, = torch.autograd.grad(weighted, prediction)
    expected = torch.ones(276)
    expected[:9] = 5  # translation and root rotation
    expected[135:147] = 5  # translation delta, rotation delta, pelvis position
    expected[210:213] = 5  # pelvis delta
    assert (expected == 5).sum() == 24
    torch.testing.assert_close(gradient, base_grad * expected)
    torch.testing.assert_close(weighted, baseline * (252 + 24 * 5) / 276)


def test_distance_map_gt_threshold_and_empty_mask_gradient():
    from moreact.losses import diffusion_motion_losses
    trans = torch.zeros(1, 4, 3)
    features = make_features(trans, torch.eye(3).expand(1, 4, 22, 3, 3),
                             torch.zeros(1, 4, 22, 3))
    history, target = features[:, :2], features[:, 2:]
    actor = target.clone()
    actor[..., JOINTS].reshape(1, 2, 22, 3)[..., 0] = 1.2
    class TranslationBody:
        def joints(self, trans, rotations, betas, genders):
            return trans[:, :, None].expand(-1, -1, 22, -1)
    prediction = target.clone()
    prediction[..., 0] = -.6  # Prediction 1.8 m, but GT 1.2 m decides the mask.
    prediction.requires_grad_()
    def evaluate(options):
        return diffusion_motion_losses(prediction, target, history, torch.zeros(276),
            torch.ones(276), TranslationBody(), torch.zeros(1, 10), torch.zeros(1, dtype=torch.long),
            {'joint_contact': 1., 'distance_map': 1.}, actor, options=options)
    _, full = evaluate({})
    _, old = evaluate({'distance_map_threshold_m': 1.})
    _, new = evaluate({'distance_map_threshold_m': 1.5})
    assert old['distance_map'] == 0 and new['distance_map'] > 0
    torch.testing.assert_close(full['distance_map'], new['distance_map'])
    assert new['joint_contact'] == old['joint_contact'] == 0
    gradient, = torch.autograd.grad(old['distance_map'], prediction, retain_graph=True)
    assert gradient.count_nonzero() == 0
    gradient, = torch.autograd.grad(new['distance_map'], prediction)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_relative_root_supervision_preserves_inherited_translation():
    from moreact.losses import diffusion_motion_losses
    from moreact.geometry import POSE, TRANSL, matrix_to_6d
    B, T = 1, 10
    trans = torch.zeros(B, T, 3)
    trans[:, :, 0] = torch.arange(T) * .02
    rotations = torch.eye(3).expand(B, T, 22, 3, 3)
    features = make_features(trans, rotations, trans[:, :, None].expand(-1, -1, 22, -1))
    true_history, target = features[:, :2], features[:, 2:]
    drifted_history = true_history.clone()
    drifted_history[..., TRANSL.start] += .3
    drifted_history[..., JOINTS.start:JOINTS.stop:3] += .3
    prediction = target.clone()
    prediction[..., TRANSL.start] += .3
    angle = torch.tensor(.35)
    c, s = angle.cos(), angle.sin()
    heading = torch.stack((torch.stack((c, -s, c*0)),
                           torch.stack((s, c, c*0)),
                           torch.tensor((0., 0., 1.))))
    heading_6d = matrix_to_6d(heading)
    drifted_history[..., POSE.start:POSE.start+6] = heading_6d
    prediction[..., POSE.start:POSE.start+6] = heading_6d
    prediction.requires_grad_()

    class TranslationBody:
        def joints(self, transl, rotations, betas, genders):
            return transl[:, :, None].expand(-1, -1, 22, -1)

    def terms(pred):
        return diffusion_motion_losses(pred, target, drifted_history,
            torch.zeros(276), torch.ones(276), TranslationBody(), torch.zeros(B, 10),
            torch.zeros(B, dtype=torch.long),
            {'root_relative_translation': 30., 'root_relative_orientation': 10.},
            target_history=true_history)[1]

    clean = terms(prediction)
    torch.testing.assert_close(clean['root_relative_translation'], torch.tensor(0.))
    torch.testing.assert_close(clean['root_relative_orientation'], torch.tensor(0.))
    incorrect = prediction.detach().clone()
    incorrect[..., TRANSL.start] += .1
    incorrect.requires_grad_()
    penalized = terms(incorrect)
    assert penalized['root_relative_translation'] > 0
    penalized['root_relative_translation'].backward()
    assert incorrect.grad[..., TRANSL].abs().sum() > 0
    wrong_turn = prediction.detach().clone()
    wrong_turn[:, 3:, POSE.start:POSE.start+6] = matrix_to_6d(torch.eye(3))
    assert terms(wrong_turn)['root_relative_orientation'] > 0
