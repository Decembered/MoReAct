"""Reproduce the fixed-seed autoregressive evaluation path and export motion arrays."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from moreact.config import dump_json
from moreact.evaluate import motion_metrics
from moreact.generate import ReactionGenerator, load_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode', default='G001T000A000R000')
    parser.add_argument('--device', default='cuda:4')
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()

    generator = ReactionGenerator(args.checkpoint, args.device)
    record, data, manifest = load_episode(generator.cfg['data']['cache'], args.episode)
    assert record['split'] == 'train' and generator.data_digest == manifest['digest']
    H, F = generator.cfg['data']['history'], generator.cfg['data']['future']
    assert args.frames % F == 0 and data['features'].shape[1] >= H + args.frames
    device = generator.device
    actor = torch.from_numpy(data['features'][0].copy()).to(device)
    truth = torch.from_numpy(data['features'][1].copy()).to(device)
    text = torch.from_numpy(data['text_embeddings'][:1].copy()).to(device)
    betas = torch.from_numpy(data['betas'].copy()).to(device)[None]
    genders = torch.from_numpy(data['genders'].copy()).to(device)[None]
    offsets = torch.from_numpy(data['offsets'].copy()).to(device)[None]
    summary = {'episode': args.episode, 'split': record['split'],
               'checkpoint': str(args.checkpoint.resolve()),
               'checkpoint_step': generator.checkpoint_step,
               'frames': args.frames, 'fps': generator.cfg['data']['fps'],
               'mode': 'autoregressive reactor history; observed actor; no GT reactor future',
               'sampling_seed_rule': 'seed*100000 + cutoff for each 8-frame segment',
               'seeds': {}}
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        history = truth[:H][None]
        blocks = []
        with torch.no_grad():
            for cutoff in range(H, H + args.frames, F):
                generator.rng.manual_seed(seed * 100000 + cutoff)
                state = generator.initialize(history, betas, genders, offsets)
                block = generator.step(actor[cutoff-H:cutoff][None], state, text)[0]
                blocks.append(block[0])
                history = block[:, -H:]
        predicted = torch.cat(blocks).cpu().numpy()
        target = truth[H:H+args.frames].cpu().numpy()
        observed_actor = actor[H:H+args.frames].cpu().numpy()
        metrics = motion_metrics(predicted, target, observed_actor,
                                 generator.cfg['data']['fps'], F, truth[H-1].cpu().numpy())
        folder = args.output / f'seed{seed}'
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'motion.npz', actor=observed_actor,
                            reactor=predicted, target=target,
                            initial_reactor=truth[:H].cpu().numpy(),
                            fps=generator.cfg['data']['fps'])
        dump_json(folder/'metadata.json', {**summary, 'seed': seed, 'metrics': metrics})
        summary['seeds'][str(seed)] = metrics
        print(seed, metrics['root_ade_m'], metrics['root_aligned_mpjpe_m'], flush=True)
    dump_json(args.output/'summary.json', summary)


if __name__ == '__main__':
    main()
