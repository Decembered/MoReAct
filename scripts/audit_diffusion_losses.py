"""Audit weighted denoiser gradients without updating the checkpoint or live run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moreact.config import dump_json
from moreact.augmentation import repair_target_boundary
from moreact.data import InterXDataset, condition_window
from moreact.generate import ReactionGenerator
from moreact.geometry import JOINTS, transform_features
from moreact.losses import diffusion_motion_losses, foot_contact_loss
from moreact.train import Trainer, batch_at, load_checkpoint, seed_all, rollout_probability


@torch.no_grad()
def vae_reference(trainer, dataset, indices):
    """GT-history posterior-mean decoded geometry: a reference, not a hard floor."""
    t = trainer
    H, F, N = (t.cfg['data'][k] for k in ('history', 'future', 'primitives'))
    rows = []
    for index, ids in enumerate(indices):
        batch = batch_at(dataset, ids, t.device)
        terms, target_foot_terms = [], []
        for k in range(N):
            start, end = k*F, k*F+H
            a, r, origin, basis = condition_window(batch['actor'][:, start:end],
                batch['reactor'][:, start:end], batch['offsets'], t.stats)
            target = transform_features(batch['reactor'][:, end:end+F], origin, basis, batch['offsets'][:, 1])
            if t.cfg.get('diffusion_loss_options', {}).get('recompute_target_deltas', False):
                target = repair_target_boundary(target, r*t.stats['std']+t.stats['mean'],
                    torch.ones(len(target), dtype=torch.bool, device=target.device))
            joints = target[..., JOINTS].reshape(len(ids), F, 22, 3)
            previous = (r[:, -1]*t.stats['std']+t.stats['mean'])[..., JOINTS].reshape(len(ids), 22, 3)
            target_foot_terms.append(foot_contact_loss(joints, joints, previous))
            target = (target-t.stats['mean'])/t.stats['std']
            pred = t.vae.decode(t.vae.encode_mean(a, r, target), a, r)
            _, values = diffusion_motion_losses(pred, target, r, t.stats['mean'], t.stats['std'],
                t.body, batch['betas'][:, 1], batch['genders'][:, 1], t.cfg['diffusion_loss'],
                actor_future=transform_features(batch['actor'][:, end:end+F], origin, basis, batch['offsets'][:, 0]),
                options=t.cfg.get('diffusion_loss_options'))
            terms.append(values)
        rows.append({'batch': index, 'target_foot_contact': float(torch.stack(target_foot_terms).mean()),
                     'terms': {k:float(torch.stack([x[k] for x in terms]).mean()) for k in terms[0]}})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:6')
    p.add_argument('--batches', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--early-checkpoint')
    args = p.parse_args()
    saved = load_checkpoint(args.checkpoint)
    g = ReactionGenerator(args.checkpoint, args.device)
    # Reuse the exact production objective without initializing an optimizer/run.
    t = object.__new__(Trainer)
    t.cfg, t.kind, t.device = g.cfg, 'diffusion', g.device
    t.vae, t.body, t.stats, t.diffusion = g.vae, g.body, g.stats, g.diffusion
    model = g.model.requires_grad_(True)
    model.load_state_dict(saved['model'])
    model.train()
    params = tuple(model.parameters())
    dataset = InterXDataset(g.cfg, 'train')
    rng = np.random.RandomState(20260926)
    indices = rng.choice(len(dataset), args.batches*args.batch_size, replace=False).reshape(args.batches, -1)
    weights = saved['config']['diffusion_loss']
    keys = list(weights)
    probability = rollout_probability(saved['step'], t.cfg['train'])
    result = {'checkpoint': str(Path(args.checkpoint).resolve()), 'step': saved['step'],
              'weights': weights, 'batch_size': args.batch_size, 'batches': args.batches,
              'train_indices': indices.tolist(), 'records': [],
              'notes': ['Raw parameter gradients of weighted terms; no optimizer step, no live run mutation.',
                        'Train weights and train-mode dropout; paired seeds across history settings.',
                        'Gradient norms are not loss shares or causal quality contributions.',
                        'Cosines/projections refer to raw summed gradients, not Adam preconditioned updates.',
                        'Full objective averages four primitives, including detached sampled history feedback.']}
    for name, prob in [('teacher', 0.), ('current_curriculum', probability), ('full_feedback', 1.)]:
        for index, ids in enumerate(indices):
            seed_all(6000+index)
            batch = batch_at(dataset, ids, g.device)
            terms = t.losses(batch, model, prob)
            gradients = []
            for key in keys:
                grads = torch.autograd.grad(terms['weighted_'+key], params, retain_graph=True, allow_unused=True)
                gradients.append(torch.cat([(v if v is not None else torch.zeros_like(p)).flatten()
                                            for p, v in zip(params, grads)]))
            G = torch.stack(gradients)
            total = G.sum(0)
            direct = torch.autograd.grad(terms['loss'], params, allow_unused=True)
            direct = torch.cat([(v if v is not None else torch.zeros_like(p)).flatten() for p,v in zip(params,direct)])
            torch.testing.assert_close(total, direct, rtol=2e-3, atol=2e-6)
            gram = G @ G.T
            norms = gram.diag().clamp_min(0).sqrt()
            total_norm = total.norm()
            dots = gram.sum(1)
            record = {'history': name, 'probability': prob, 'batch': index,
                      'loss': float(terms['loss'].detach()), 'total_gradient_norm': float(total_norm),
                      'gradient_sum_max_abs_error': float((total-direct).abs().max()), 'terms': {}}
            for i, key in enumerate(keys):
                record['terms'][key] = {'raw': float(terms[key].detach()),
                    'weighted': float(terms['weighted_'+key].detach()),
                    'weighted_gradient_norm': float(norms[i]),
                    'cosine_total': float(dots[i]/(norms[i]*total_norm).clamp_min(1e-30)),
                    'cosine_latent': float(gram[i, 0]/(norms[i]*norms[0]).clamp_min(1e-30)),
                    'projection_total': float(dots[i]/total_norm.square().clamp_min(1e-30))}
            result['records'].append(record)
            del terms, gradients, G, total, direct, grads, gram
            dump_json(args.output, result)
            print(name, index, 'complete', flush=True)
    if args.early_checkpoint:
        validation = InterXDataset(g.cfg, 'val')
        val_ids = np.random.RandomState(20260927).choice(len(validation), args.batches*args.batch_size,
                                                      replace=False).reshape(args.batches, -1)
        result['paired_validation'] = {'indices': val_ids.tolist(), 'records': []}
        result['validation_log_coverage'] = {
            'first_512_windows_episodes': sorted({validation.records[validation.windows[i][0]]['episode']
                                                for i in range(min(512, len(validation)))}),
            'total_episodes': len(validation.records), 'total_windows': len(validation),
            'probe_episodes': len({validation.windows[int(i)][0] for i in val_ids.flatten()})}
        for checkpoint in [args.early_checkpoint, args.checkpoint]:
            snap = load_checkpoint(checkpoint)
            for key, value in t.vae.state_dict().items():
                torch.testing.assert_close(value.cpu(), snap['vae'][key], rtol=0, atol=0)
            for key, value in t.stats.items():
                torch.testing.assert_close(value.cpu(), snap['stats'][key], rtol=0, atol=0)
            model.load_state_dict(snap['ema'])
            model.eval()
            for name, prob in [('teacher', 0.), ('full_feedback', 1.)]:
                for index, ids in enumerate(val_ids):
                    seed_all(9000+index)
                    with torch.no_grad():
                        terms = t.losses(batch_at(validation, ids, g.device), model, prob)
                    result['paired_validation']['records'].append({'step': snap['step'], 'history': name,
                        'batch': index, 'terms': {k:float(v) for k,v in terms.items()}})
            print('paired_validation',snap['step'],'complete',flush=True)
            dump_json(args.output, result)
        result['vae_teacher_reference'] = vae_reference(t, validation, val_ids)
        dump_json(args.output, result)
    print('Done', flush=True)


if __name__ == '__main__':
    main()
