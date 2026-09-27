"""Quantify how the diffusion objective responds to accumulated root drift.

Accumulated rollout drift is a rigid offset of the reactor history relative to
the ground truth, because the anchor frame is built from that history. This tool
injects a controlled offset in the episode frame and measures, using the
production objective functions:

  ideal     loss terms for a frozen non-adaptive predictor: the GT local motion
            expressed in the undrifted anchor frame, scored against the drifted
            target. Isolates the loss geometry from the model.
  model     trained model (EMA) conditioned on the drifted history, i.e. the
            rollout path: loss terms plus its catch-up behaviour.
  grad      per-term parameter gradients of the training objective and their
            pairwise cosines at selected offsets (raw weights, train mode).
  feedback  drift actually accumulated by the real 4-primitive feedback loop,
            compared with the drift seen at evaluation time.

The offset has two independent knobs measured from the last history frame
pelvis: a translation (metres, along the GT travel direction of the window) and
a yaw rotation (degrees).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moreact.config import dump_json
from moreact.data import InterXDataset, condition_window
from moreact.generate import ReactionGenerator
from moreact.geometry import (DJOINTS, DROT, DTRANS, JOINTS, POSE, TRANSL,
                              matrix_to_6d, rotation_6d_to_matrix, transform_features)
from moreact.losses import diffusion_motion_losses
from moreact.train import batch_at, load_checkpoint, reconstruction_mse, seed_all


def yaw_matrix(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((torch.stack((c, -s, z), -1), torch.stack((s, c, z), -1),
                        torch.stack((z, z, o), -1)), -2)


def yaw_of(joints):
    hips = joints[..., 2, :] - joints[..., 1, :]
    return torch.atan2(hips[..., 1], hips[..., 0])


def drift_history(features, delta_xy, yaw_deg):
    """Rigidly offset a reactor history in the episode frame.

    Deltas are world-frame vectors and the relative root rotation conjugates under
    a global yaw, so both channels are transformed analytically and stay
    consistent with what the rollout feedback would produce.
    """
    B, H = features.shape[:2]
    out = features.clone()
    trans = features[..., TRANSL]
    joints = features[..., JOINTS].reshape(B, H, 22, 3)
    pivot = joints[:, -1, 0]
    R = yaw_matrix(yaw_deg * math.pi / 180)
    trans = torch.einsum('bij,btj->bti', R, trans - pivot[:, None]) + pivot[:, None]
    joints = torch.einsum('bij,btj->bti', R, (joints - pivot[:, None, None]).reshape(B, H * 22, 3)
                          ).reshape(B, H, 22, 3) + pivot[:, None, None]
    poses = R[:, None, None] @ rotation_6d_to_matrix(features[..., POSE].reshape(B, H, 22, 6))
    delta = R[:, None] @ rotation_6d_to_matrix(features[..., DROT]) @ R[:, None].transpose(-1, -2)
    dtrans = torch.einsum('bij,btj->bti', R, features[..., DTRANS])
    djoints = torch.einsum('bij,btj->bti', R, features[..., DJOINTS].reshape(B, H * 22, 3))
    shift = torch.cat((delta_xy, torch.zeros_like(delta_xy[:, :1])), -1)[:, None]
    out[..., TRANSL] = trans + shift
    out[..., JOINTS] = (joints + shift[:, :, None]).reshape(B, H, 66)
    out[..., POSE] = matrix_to_6d(poses).flatten(-2)
    out[..., DROT] = matrix_to_6d(delta)
    out[..., DTRANS] = dtrans
    out[..., DJOINTS] = djoints.reshape(B, H, 66)
    return out


class Probe:
    def __init__(self, checkpoint, device, batches, batch_size, seed):
        self.gen = ReactionGenerator(checkpoint, device)
        self.checkpoint = checkpoint
        cfg = self.gen.cfg
        self.device = self.gen.device
        self.dataset = InterXDataset(cfg, 'train')
        rng = np.random.RandomState(seed)
        self.indices = rng.choice(len(self.dataset), batches * batch_size,
                                  replace=False).reshape(batches, -1)
        self.H, self.F, self.N = cfg['data']['history'], cfg['data']['future'], cfg['data']['primitives']
        self.latent_dim = cfg['model']['latent_dim']
        self.mean, self.std = self.gen.stats['mean'], self.gen.stats['std']

    def batch(self, ids):
        return batch_at(self.dataset, ids, self.device)

    def travel_direction(self, batch):
        trans = batch['reactor'][..., TRANSL]
        delta = trans[:, self.H + self.F - 1, :2] - trans[:, self.H - 1, :2]
        norm = delta.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(delta)
        fallback[:, 0] = 1
        return torch.where(norm > 1e-6, delta / norm.clamp_min(1e-9), fallback)

    def primitive0(self, batch, drifted):
        actor = batch['actor'][:, :self.H]
        gt = batch['reactor'][:, :self.H + self.F]
        a, r, origin_d, basis_d = condition_window(actor, drifted, batch['offsets'], self.gen.stats)
        _, _, origin_0, basis_0 = condition_window(actor, gt[:, :self.H], batch['offsets'],
                                                   self.gen.stats)
        future = gt[:, self.H:self.H + self.F]
        offset = batch['offsets'][:, 1]
        target = transform_features(future, origin_d, basis_d, offset)
        target = (target - self.mean) / self.std
        ideal = transform_features(future, origin_0, basis_0, offset)
        ideal = (ideal - self.mean) / self.std
        return a, r, (origin_d, basis_d), target, ideal

    def terms(self, batch, prediction, target, history, anchor):
        origin, basis = anchor
        actor_future = transform_features(batch['actor'][:, self.H:self.H + self.F], origin, basis,
                                          batch['offsets'][:, 0]).detach()
        weights = self.gen.cfg['diffusion_loss']
        total, values = diffusion_motion_losses(prediction, target, history, self.mean, self.std,
                                               self.gen.body, batch['betas'][:, 1],
                                               batch['genders'][:, 1], weights,
                                               actor_future=actor_future)
        return float(total), {k: float(v) for k, v in values.items()}

    def sample_prediction(self, batch, a, r, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        z = self.gen.diffusion.sample(self.gen.model.eval(), a, r, batch['text'],
                                      self.latent_dim, 1.0, generator)
        return self.gen.vae.decode(z * self.gen.vae.latent_scale, a, r)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:5')
    parser.add_argument('--batches', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--parts', default='ideal,model,feedback')
    parser.add_argument('--translations', type=float, nargs='*',
                        default=[0.0, 0.05, 0.10, 0.20, 0.30, 0.50])
    parser.add_argument('--yaws', type=float, nargs='*', default=[0.0, 10.0, 20.0])
    parser.add_argument('--grad-translations', type=float, nargs='*', default=[0.0, 0.20])
    args = parser.parse_args()
    seed_all(args.seed)

    probe = Probe(args.checkpoint, args.device, args.batches, args.batch_size, args.seed)
    defaults = probe.gen.model
    result = {
        'checkpoint': str(Path(args.checkpoint).resolve()), 'device': args.device,
        'batches': args.batches, 'batch_size': args.batch_size, 'seed': args.seed,
        'notes': [
            'Offsets are rigid history shifts: translation along the GT travel direction of the '
            'window plus yaw about the pelvis of the last history frame.',
            'prediction=ideal is the GT local motion expressed in the undrifted anchor frame, i.e. '
            'a predictor that does not react to the drift.',
            'model/feedback use the generator EMA weights and the production sampling path; grad '
            'uses raw training weights in train mode.',
            'Terms are the production diffusion_motion_losses with the checkpoint weights.']}
    parts = set(args.parts.split(','))
    zero = None

    for index, ids in enumerate(probe.indices):
        batch = probe.batch(ids)
        direction = probe.travel_direction(batch)
        zero = torch.zeros_like(direction[:, 0])

        if 'ideal' in parts:
            for delta in args.translations:
                for yaw in (args.yaws if delta == 0 else [0.0]):
                    moved = drift_history(batch['reactor'][:, :probe.H], direction * delta,
                                          torch.full_like(zero, yaw))
                    a, r, anchor, target, ideal = probe.primitive0(batch, moved)
                    total, values = probe.terms(batch, ideal, target, r, anchor)
                    gap = (ideal - target)[..., :3].mul(probe.std[:3]).norm(dim=-1).mean()
                    result.setdefault('ideal', []).append({
                        'batch': index, 'delta_m': delta, 'yaw_deg': yaw,
                        'local_root_gap_cm': float(gap * 100), 'total': total,
                        'weighted': {k: probe.gen.cfg['diffusion_loss'][k] * v
                                     for k, v in values.items()}, 'raw': values})

        if 'model' in parts:
            for delta in args.translations:
                moved = drift_history(batch['reactor'][:, :probe.H], direction * delta, zero)
                a, r, anchor, target, _ = probe.primitive0(batch, moved)
                with torch.no_grad():
                    prediction = probe.sample_prediction(batch, a, r, args.seed + index)
                total, values = probe.terms(batch, prediction, target, r, anchor)
                physical = prediction * probe.std + probe.mean
                world = transform_features(physical, *anchor, batch['offsets'][:, 1], inverse=True)
                first_step = (world[:, 0, TRANSL] - moved[:, -1, TRANSL]).norm(dim=-1).mean() * 100
                gt_step = (batch['reactor'][:, probe.H, TRANSL]
                           - batch['reactor'][:, probe.H - 1, TRANSL]).norm(dim=-1).mean() * 100
                result.setdefault('model', []).append({
                    'batch': index, 'delta_m': delta, 'total': total,
                    'first_step_cm': float(first_step), 'gt_first_step_cm': float(gt_step),
                    'weighted': {k: probe.gen.cfg['diffusion_loss'][k] * v
                                 for k, v in values.items()}, 'raw': values})

        if 'feedback' in parts:
            context = batch['reactor'].clone()
            for k in range(probe.N - 1):
                start, end = k * probe.F, k * probe.F + probe.H
                a, r, origin, basis = condition_window(batch['actor'][:, start:end],
                                                       context[:, start:end],
                                                       batch['offsets'], probe.gen.stats)
                with torch.no_grad():
                    pred = probe.sample_prediction(batch, a, r, args.seed + 100 + k)
                    local = pred * probe.std + probe.mean
                    world = transform_features(local, origin, basis, batch['offsets'][:, 1],
                                               inverse=True)
                    world = probe.gen.body.repair(world, context[:, start:end],
                                                  batch['betas'][:, 1], batch['genders'][:, 1])
                gt = batch['reactor'][:, end:end + probe.F]
                gt_local = transform_features(gt, origin, basis, batch['offsets'][:, 1])
                gap = world[:, -1, TRANSL] - gt[:, -1, TRANSL]
                yaw_gap = yaw_of(world[..., JOINTS].reshape(len(ids), probe.F, 22, 3)[:, -1]) - \
                    yaw_of(gt[..., JOINTS].reshape(len(ids), probe.F, 22, 3)[:, -1])
                yaw_gap = torch.atan2(torch.sin(yaw_gap), torch.cos(yaw_gap)) * 180 / math.pi
                result.setdefault('feedback', []).append({
                    'batch': index, 'step': k + 1,
                    'pelvis_gap_cm': float(gap.norm(dim=-1).mean() * 100),
                    'pelvis_gap_xy_cm': float(gap[:, :2].norm(dim=-1).mean() * 100),
                    'yaw_gap_deg': float(yaw_gap.abs().mean()),
                    'loss_visible_local_root_err_cm': float(
                        (local[..., TRANSL] - gt_local[..., TRANSL]).norm(dim=-1).mean() * 100)})
                context[:, end:end + probe.F] = world
        dump_json(args.output, result)
        print('batch', index, 'done', flush=True)

    if 'grad' in parts:
        saved = load_checkpoint(args.checkpoint)
        probe.gen.model.load_state_dict(saved['model'])
        model = probe.gen.model.train().requires_grad_(True)
        params = tuple(model.parameters())
        keys = list(probe.gen.cfg['diffusion_loss'])
        batch = probe.batch(probe.indices[0])
        direction = probe.travel_direction(batch)
        for delta in args.grad_translations:
            moved = drift_history(batch['reactor'][:, :probe.H], direction * delta, zero)
            a, r, anchor, target, _ = probe.primitive0(batch, moved)
            seed_all(args.seed + 7)
            with torch.no_grad():
                z, _, _ = probe.gen.vae.encode(a, r, target)
                z = z / probe.gen.vae.latent_scale
                step = torch.randint(probe.gen.diffusion.steps, (len(z),), device=probe.device)
                noisy = probe.gen.diffusion.corrupt(z, step, torch.randn_like(z))
            predicted_z = model(noisy, step, a, r, batch['text'])
            prediction = probe.gen.vae.decode(predicted_z * probe.gen.vae.latent_scale, a, r)
            weights = probe.gen.cfg['diffusion_loss']
            actor_future = transform_features(batch['actor'][:, probe.H:probe.H + probe.F],
                                              *anchor, batch['offsets'][:, 0]).detach()
            motion_total, values = diffusion_motion_losses(
                prediction, target, r, probe.mean, probe.std, probe.gen.body,
                batch['betas'][:, 1], batch['genders'][:, 1], weights, actor_future=actor_future)
            terms = dict(values)
            terms['latent_mse'] = reconstruction_mse(predicted_z, z)
            total = weights['latent_mse'] * terms['latent_mse'] + motion_total
            gradients = []
            for key in keys:
                tensor = weights[key] * terms[key]
                grads = torch.autograd.grad(tensor, params, retain_graph=True, allow_unused=True)
                gradients.append(torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                            for p, g in zip(params, grads)]))
            G = torch.stack(gradients)
            direct = torch.autograd.grad(total, params, allow_unused=True)
            direct = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                for p, g in zip(params, direct)])
            norms = G.norm(dim=1)
            cosines = {}
            for i, key in enumerate(keys):
                for j in range(i + 1, len(keys)):
                    denom = (norms[i] * norms[j]).clamp_min(1e-30)
                    cosines[key + '|' + keys[j]] = float(G[i] @ G[j] / denom)
            result.setdefault('grad', []).append({
                'delta_m': delta, 'total': float(total.detach()),
                'total_gradient_norm': float(direct.norm()),
                'gradient_sum_max_abs_error': float((G.sum(0) - direct).abs().max()),
                'term_gradient_norm': {k: float(norms[i]) for i, k in enumerate(keys)},
                'term_gradient_share': {k: float(norms[i] / direct.norm()) for i, k in enumerate(keys)},
                'cosine': cosines})
            del gradients, G
        probe.gen.model = defaults
        dump_json(args.output, result)
        print('grad done', flush=True)
    dump_json(args.output, result)

    print()
    print('%-46s %10s %10s' % ('ideal: delta / yaw', 'root_pos', 'joint_vel'))
    for row in result.get('ideal', []):
        if row['batch']:
            continue
        print('%-46s %10.4f %10.4f' % (
            'delta %.2fm yaw %.0fdeg' % (row['delta_m'], row['yaw_deg']),
            row['weighted'].get('root_position', float('nan')),
            row['weighted'].get('joint_velocity', float('nan'))))
    print('written', args.output)


if __name__ == '__main__':
    main()
