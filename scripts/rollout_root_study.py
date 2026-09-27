"""Paired 120-frame rollout study with optional one-step history interventions."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moreact.config import dump_json
from moreact.evaluate import motion_metrics
from moreact.generate import ReactionGenerator, load_episode
from moreact.geometry import JOINTS, TRANSL, transform_features


def select(cache, split, seed, frames=120, history=2):
    records = json.loads((Path(cache) / 'manifest.json').read_text())['records']
    groups = {}
    for record in records:
        if record['split'] == split and record['frames'] >= frames + history:
            action = re.search(r'A(\d{3})', record['episode']).group(1)
            groups.setdefault(action, []).append(record)
    if len(groups) != 40:
        raise ValueError('Expected 40 eligible action classes, got ' + str(len(groups)))
    rng = np.random.RandomState(seed)
    return [groups[key][int(rng.randint(len(groups[key])))] for key in sorted(groups)]


def yaw(history):
    joints = history[..., JOINTS].reshape(*history.shape[:-1], 22, 3)
    hip = joints[..., 2, :] - joints[..., 1, :]
    return torch.atan2(hip[..., 1], hip[..., 0])


def move(history, offset, shift=(0., 0., 0.), angle=0.):
    pelvis = history[:, -1, JOINTS].reshape(-1, 22, 3)[:, 0]
    c, s = math.cos(angle), math.sin(angle)
    rotation = history.new_tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]).expand(len(history), -1, -1)
    origin = pelvis - torch.einsum('bij,bj->bi', rotation, pelvis) + history.new_tensor(shift)[None]
    return transform_features(history, origin, rotation, offset, inverse=True)


def score_block(pred, target, history, true_history):
    p = pred[0, :, JOINTS].reshape(-1, 22, 3)[:, 0]
    q = target[0, :, JOINTS].reshape(-1, 22, 3)[:, 0]
    previous = history[0, -1, JOINTS].reshape(22, 3)[0]
    true_previous = true_history[0, -1, JOINTS].reshape(22, 3)[0]
    displacement = (p[-1] - previous) - (q[-1] - true_previous)
    direction = q[-1, :2] - true_previous[:2]
    direction = direction / direction.norm().clamp_min(1e-8)
    angle = torch.atan2(torch.sin(yaw(pred) - yaw(target)),
                        torch.cos(yaw(pred) - yaw(target)))
    return {'root_ade_m': float((p-q).norm(dim=-1).mean()),
            'root_fde_m': float((p[-1]-q[-1]).norm()),
            'along_track_delta_error_m': float((displacement[:2]*direction).sum()),
            'heading_error_deg': float(angle.abs().mean()*180/math.pi),
            'boundary_step_m': float((p[0]-previous).norm()),
            'boundary_velocity_error_m': float(((p[0]-previous)-(q[0]-true_previous)).norm())}


@torch.no_grad()
def evaluate_one(generator, data, seed, frames, intervene=False):
    device = generator.device
    actor = torch.from_numpy(data['features'][0].copy()).to(device)
    truth = torch.from_numpy(data['features'][1].copy()).to(device)
    text = torch.from_numpy(data['text_embeddings'][:1].copy()).to(device)
    betas = torch.from_numpy(data['betas'].copy()).to(device)[None]
    genders = torch.from_numpy(data['genders'].copy()).to(device)[None]
    offsets = torch.from_numpy(data['offsets'].copy()).to(device)[None]
    H, F = generator.cfg['data']['history'], generator.cfg['data']['future']
    if frames % F:
        raise ValueError('Study requires full primitives')
    generated, teacher, blocks, probes = [], [], [], []
    history = truth[:H][None]
    for end in range(H, H + frames, F):
        ah = actor[end-H:end][None]
        true_history = truth[end-H:end][None]
        target = truth[end:end+F][None]
        segment_seed = seed*100000 + end

        def predict(rh):
            generator.rng.manual_seed(segment_seed)
            state = generator.initialize(rh, betas, genders, offsets)
            return generator.step(ah, state, text)[0]

        block = predict(history)
        teacher_block = predict(true_history)
        blocks.append({'cutoff': end, **score_block(block, target, history, true_history)})
        if intervene and end in (H+40, H+80):
            predicted_pelvis = history[0, -1, JOINTS.start:JOINTS.start+3]
            true_pelvis = true_history[0, -1, JOINTS.start:JOINTS.start+3]
            shift = (true_pelvis - predicted_pelvis).tolist()
            angle = float(torch.atan2(torch.sin(yaw(true_history)[0, -1] - yaw(history)[0, -1]),
                                      torch.cos(yaw(true_history)[0, -1] - yaw(history)[0, -1])))
            state_at_drift = move(true_history, offsets[:, 1],
                                  (predicted_pelvis-true_pelvis).tolist(),
                                  -angle)
            versions = {'original': history,
                        'root_reset': move(history, offsets[:, 1], shift),
                        'heading_reset': move(history, offsets[:, 1], angle=angle),
                        'gt_local_state_at_drift': state_at_drift}
            for mode, rh in versions.items():
                pred = block if mode == 'original' else predict(rh)
                probes.append({'cutoff': end, 'mode': mode,
                               **score_block(pred, target, rh, true_history)})
        generated.append(block[0]); teacher.append(teacher_block[0])
        history = block[:, -H:]

    generated = torch.cat(generated).cpu().numpy()
    teacher = torch.cat(teacher).cpu().numpy()
    gt = truth[H:H+frames].cpu().numpy()
    actor_future = actor[H:H+frames].cpu().numpy()
    previous = truth[H-1].cpu().numpy()
    metrics = {key: motion_metrics(pred, gt, actor_future, generator.cfg['data']['fps'], F, previous)
               for key, pred in [('feedback', generated), ('teacher', teacher)]}
    return {'metrics': metrics, 'blocks': blocks, 'probes': probes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', choices=('train','val','test'), default='test')
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--device', default='cuda:7')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2])
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--intervene-on', help='Checkpoint basename for one-step probes')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    first = ReactionGenerator(args.checkpoints[0], args.device)
    cache = first.cfg['data']['cache']
    if args.selection and args.selection.exists():
        records = json.loads(args.selection.read_text())['records']
    else:
        records = select(cache, args.split, 20260927, args.frames, first.cfg['data']['history'])
        dump_json(args.selection or args.output/'selection.json',
                  {'split': args.split, 'seed': 20260927, 'records': records})
    del first
    out = args.output/'per_episode.jsonl'
    existing = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    done = {(row['checkpoint'], row['episode'], row['seed']) for row in existing}
    for checkpoint in args.checkpoints:
        generator = ReactionGenerator(checkpoint, args.device)
        if generator.data_digest != json.loads((Path(cache)/'manifest.json').read_text())['digest']:
            raise ValueError('Checkpoint/data digest mismatch')
        for record in records:
            _, data, _ = load_episode(cache, record['episode'])
            for seed in args.seeds:
                key = (str(Path(checkpoint).resolve()), record['episode'], seed)
                if key in done:
                    continue
                result = evaluate_one(generator, data, seed, args.frames,
                    intervene=Path(checkpoint).name == args.intervene_on)
                row = {'checkpoint': key[0], 'step': generator.checkpoint_step,
                       'episode': record['episode'], 'action': re.search(r'A(\d{3})', record['episode']).group(1),
                       'split': record['split'], 'seed': seed, **result}
                with out.open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                done.add(key)
            print(checkpoint, record['episode'], 'complete', flush=True)
        del generator
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
