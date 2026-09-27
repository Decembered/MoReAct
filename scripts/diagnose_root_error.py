"""Paired root-error attribution using the VAE embedded in one diffusion checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moreact.config import dump_json
from moreact.data import condition_window
from moreact.generate import ReactionGenerator, load_episode
from moreact.geometry import JOINTS, TRANSL, transform_features


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def measure(g, data, mode, seed, frames):
    H, F = g.cfg['data']['history'], g.cfg['data']['future']
    device = g.device
    features = torch.as_tensor(data['features'], device=device)
    offsets = torch.as_tensor(data['offsets'], device=device)[None]
    betas = torch.as_tensor(data['betas'], device=device)[None]
    genders = torch.as_tensor(data['genders'], device=device)[None]
    text = torch.as_tensor(data['text_embeddings'][:1], device=device)
    frames = min(frames, features.shape[1] - H)
    rng = torch.Generator(device=device).manual_seed(seed)
    rh = features[1, :H][None].clone()
    predictions, raw_translations = [], []
    for end in range(H, H + frames, F):
        if mode.endswith('_teacher'):
            rh = features[1, end-H:end][None]
        ah = features[0, end-H:end][None]
        a, r, origin, basis = condition_window(ah, rh, offsets, g.stats)
        if mode.startswith('vae_'):
            target = features[1, end:min(end+F, features.shape[1])][None]
            if target.shape[1] < F:
                target = torch.cat((target, target[:, -1:].expand(-1, F-target.shape[1], -1)), 1)
            local = transform_features(target, origin, basis, offsets[:, 1])
            normalized = (local-g.stats['mean'])/g.stats['std']
            if mode == 'vae_sample_teacher':
                mu, logvar = g.vae._encode_distribution(a, r, normalized)
                z = mu + (.5*logvar).exp()*torch.randn(mu.shape, device=device, generator=rng)
            else:
                z = g.vae.encode_mean(a, r, normalized)
        else:
            z = g.diffusion.sample(g.model, a, r, text, g.cfg['model']['latent_dim'],
                                   guidance=1., generator=rng)*g.vae.latent_scale
        pred = g.vae.decode(z, a, r)
        world = transform_features(pred*g.stats['std']+g.stats['mean'], origin, basis,
                                   offsets[:, 1], inverse=True)
        take = min(F, H+frames-end)
        raw_translations.append(world[0, :take, TRANSL].clone())
        world = g.body.repair(world, rh, betas[:, 1], genders[:, 1])
        predictions.append(world[0, :take])
        rh = torch.cat((rh, world), 1)[:, -H:]
    pred = torch.cat(predictions)
    target = features[1, H:H+frames]
    pj = pred[:, JOINTS].reshape(-1, 22, 3)
    tj = target[:, JOINTS].reshape(-1, 22, 3)
    delta = pj[:, 0]-tj[:, 0]
    error = delta.norm(dim=-1)
    translation_error = (pred[:, TRANSL]-target[:, TRANSL]).norm(dim=-1)
    return {
        'frames': frames,
        'root_ade_m': error.mean().item(), 'root_fde_m': error[-1].item(),
        'root_xy_ade_m': delta[:, :2].norm(dim=-1).mean().item(),
        'root_z_mae_m': delta[:, 2].abs().mean().item(),
        'translation_ade_m': translation_error.mean().item(),
        'repair_translation_change_max_m': (torch.cat(raw_translations)-pred[:, TRANSL]).abs().max().item(),
        'world_mpjpe_m': (pj-tj).norm(dim=-1).mean().item(),
        'first_primitive_root_ade_m': error[:F].mean().item(),
        'last_primitive_root_ade_m': error[-F:].mean().item(),
        'root_error_by_frame_m': error.cpu().tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--limit', type=int, default=8)
    p.add_argument('--frames', type=int, default=192)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--episodes', nargs='*', default=[])
    args = p.parse_args()
    g = ReactionGenerator(args.checkpoint, args.device)
    cache = Path(g.cfg['data']['cache'])
    manifest = json.loads((cache/'manifest.json').read_text())
    if manifest['digest'] != g.data_digest:
        raise ValueError('Checkpoint/data mismatch')
    candidates = [r for r in manifest['records'] if r['split'] == 'val'
                  and r['frames'] > g.cfg['data']['history']]
    indices = np.linspace(0, len(candidates)-1, min(args.limit, len(candidates)), dtype=int)
    records = [candidates[i] for i in indices]
    seen = {r['episode'] for r in records}
    for episode in args.episodes:
        if episode not in seen:
            records.append(next(r for r in manifest['records'] if r['episode'] == episode))
            seen.add(episode)
    modes = ['vae_mean_teacher', 'vae_sample_teacher', 'vae_mean_oracle_feedback',
             'diffusion_teacher', 'diffusion_feedback']
    result = {
        'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_step': g.checkpoint_step,
        'checkpoint_sha256': checkpoint_digest(args.checkpoint),
        'data_digest': g.data_digest, 'device': args.device, 'frames_limit': args.frames,
        'seeds': args.seeds, 'guidance': 1., 'config': g.cfg,
        'notes': [
            'Same embedded frozen VAE, EMA denoiser, normalization, frames and FK repair in all arms.',
            'Teacher arms reset reactor history to GT every primitive.',
            'VAE arms use GT future in the encoder; oracle feedback is diagnostic, not causal generation.',
            'Diffusion arms never use target future as a model input; teacher/feedback use paired RNG streams.',
            'Errors to one recorded future are reference metrics, not unique causal or generative quality attribution.',
            'Validation episodes selected evenly across manifest order; named test examples reported separately.',
        ], 'records': [],
    }
    for record in records:
        _, data, _ = load_episode(cache, record['episode'])
        for mode in modes:
            seeds = [args.seeds[0]] if mode in ('vae_mean_teacher', 'vae_mean_oracle_feedback') else args.seeds
            for seed in seeds:
                metrics = measure(g, data, mode, seed, args.frames)
                result['records'].append({'episode': record['episode'], 'split': record['split'],
                                          'mode': mode, 'seed': seed, **metrics})
        print(record['episode']+' complete', flush=True)
        dump_json(args.output, result)
    result['aggregate'] = {}
    for split in sorted({r['split'] for r in records}):
        result['aggregate'][split] = {}
        for mode in modes:
            rows = [r for r in result['records'] if r['split'] == split and r['mode'] == mode]
            keys = [k for k in rows[0] if k.endswith('_m') and k != 'root_error_by_frame_m']
            result['aggregate'][split][mode] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    dump_json(args.output, result)
    print(json.dumps(result['aggregate'], indent=2), flush=True)


if __name__ == '__main__':
    main()
