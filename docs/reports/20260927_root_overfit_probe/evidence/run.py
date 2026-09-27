"""Single-episode fixed-window memorization probe; temporary study script."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from moreact.data import InterXDataset
from moreact.generate import ReactionGenerator, load_episode
from moreact.train import Trainer, batch_at, load_checkpoint
from rollout_root_study import evaluate_one


SOURCE = ROOT / 'runs/diffusion_stage2_b512_stratified_20260926/step_027000.pt'
OUT = Path(__file__).resolve().parent
DEVICE = 'cuda:0'
UPDATES = 300
BATCH = 2


def select_episode(cfg):
    ds = InterXDataset(cfg, 'train')
    for ep, rec in enumerate(ds.records):
        if rec['frames'] >= 122 and len(rec.get('episode', '')):
            starts = list(range(0, 89, 8))
            lookup = {window: i for i, window in enumerate(ds.windows)}
            if all((ep, start) in lookup for start in starts):
                return rec['episode'], [lookup[(ep, start)] for start in starts]
    raise RuntimeError('No train episode with 120-frame coverage')


def evaluate(path, episode, seeds=(0, 1, 2)):
    generator = ReactionGenerator(path, DEVICE)
    _, data, _ = load_episode(generator.cfg['data']['cache'], episode)
    rows = [evaluate_one(generator, data, seed, 120) for seed in seeds]
    output = {}
    for mode in ('teacher', 'feedback'):
        keys = ('root_ade_m', 'root_fde_m', 'root_aligned_mpjpe_m',
                'pair_root_distance_error_m', 'boundary_velocity_jump_mps')
        output[mode] = {key: float(np.mean([row['metrics'][mode][key] for row in rows]))
                        for key in keys}
    del generator
    torch.cuda.empty_cache()
    return output


def main():
    saved = load_checkpoint(SOURCE)
    cfg = copy.deepcopy(saved['config'])
    cfg['train']['device'] = DEVICE
    cfg['train']['batch_size'] = BATCH
    cfg['train']['threads'] = 1
    cfg['train']['learning_rate'] = 1e-4
    cfg['train']['ema'] = 0.95
    cfg['history_augmentation'] = {'probability': 0., 'std_m': 0., 'max_m': 0.05}
    cfg['diffusion_loss_options'] = {'feature_root_weight': 5.,
        'distance_map_threshold_m': 1.5, 'recompute_target_deltas': True}
    episode, indices = select_episode(cfg)
    result = {'source': str(SOURCE), 'episode': episode, 'window_indices': indices,
              'window_starts': list(range(0, 89, 8)), 'batch_size': BATCH,
              'updates': UPDATES, 'device': DEVICE, 'seed': 20260927,
              'learning_rate': cfg['train']['learning_rate'], 'ema': cfg['train']['ema'],
              'config': cfg, 'evaluations': {}}
    (OUT / 'selection.json').write_text(json.dumps({k: v for k, v in result.items()
        if k != 'config'}, indent=2) + '\n')
    result['evaluations']['source'] = evaluate(SOURCE, episode)
    print('baseline', result['evaluations']['source'], flush=True)
    trainer = Trainer(cfg, 'diffusion', OUT, resume=SOURCE, new_objective=True,
        resume_overrides=('batch_size', 'learning_rate', 'ema'))
    for group in trainer.optimizer.param_groups:
        group['lr'] = cfg['train']['learning_rate']
    trainer.model.train()
    rng = np.random.RandomState(20260927)
    for update in range(1, UPDATES + 1):
        chosen = rng.choice(indices, size=BATCH, replace=False)
        batch = batch_at(trainer.dataset, chosen, trainer.device)
        trainer.optimizer.zero_grad(set_to_none=True)
        losses = trainer.losses(batch, trainer.model, probability=1.)
        if not torch.isfinite(losses['loss']):
            raise FloatingPointError(f'Nonfinite loss at update {update}')
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), cfg['train']['grad_clip'])
        trainer.optimizer.step()
        with torch.no_grad():
            for dst, src in zip(trainer.ema.parameters(), trainer.model.parameters()):
                dst.lerp_(src, 1-cfg['train']['ema'])
        trainer.step += 1
        if update in (1, 50, 100, 200, 300):
            line = {'update': update, 'checkpoint_step': trainer.step,
                    **{key: float(value.detach()) for key, value in losses.items()}}
            with (OUT / 'losses.jsonl').open('a') as log:
                log.write(json.dumps(line) + '\n')
            print('train', update, line['loss'], flush=True)
        del batch, losses
        if update in (100, 300):
            checkpoint_name = f'overfit_{update}.pt'
            trainer.checkpoint(checkpoint_name)
            result['evaluations'][str(update)] = evaluate(OUT / checkpoint_name, episode)
            print('eval', update, result['evaluations'][str(update)], flush=True)
            (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
