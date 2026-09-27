"""Evaluate selected immutable diffusion checkpoints on fixed 120-frame rollouts."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moreact.config import dump_json


METRICS = ('root_ade_m', 'root_fde_m', 'root_aligned_mpjpe_m',
           'pair_root_distance_error_m', 'boundary_velocity_jump_mps')


def score(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    episodes = {}
    for row in rows:
        episodes.setdefault(row['episode'], []).append(row['metrics']['feedback'])
    if len(rows) != 120 or len(episodes) != 40 or any(len(values) != 3 for values in episodes.values()):
        raise ValueError(f'Incomplete fixed rollout evaluation: {path}')
    return {metric: sum(sum(item[metric] for item in values) / len(values)
                        for values in episodes.values()) / len(episodes)
            for metric in METRICS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--launcher-pid', type=int, required=True)
    parser.add_argument('--device', default='cuda:7')
    parser.add_argument('--steps', nargs='+', type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / 'status.json'
    results = {}
    try:
        for step in args.steps:
            checkpoint = args.run / f'step_{step:06d}.pt'
            while not checkpoint.is_file():
                if not Path(f'/proc/{args.launcher_pid}/cmdline').exists():
                    raise RuntimeError(f'Launcher exited before checkpoint {step}')
                dump_json(status_path, {'state': 'waiting', 'next_step': step,
                                        'completed': results, 'launcher_pid': args.launcher_pid})
                time.sleep(30)
            folder = args.output / f'step_{step:06d}'
            command = [sys.executable, '-u', str(ROOT / 'scripts/rollout_root_study.py'),
                       '--checkpoints', str(checkpoint), '--output', str(folder),
                       '--split', args.split, '--selection', str(args.selection),
                       '--device', args.device, '--frames', '120', '--seeds', '0', '1', '2']
            dump_json(status_path, {'state': 'evaluating', 'step': step,
                                    'completed': results, 'command': command})
            with (args.output / f'step_{step:06d}.log').open('a') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, 'OMP_NUM_THREADS': '1'}, check=True)
            result = score(folder / 'per_episode.jsonl')
            results[str(step)] = result
            dump_json(args.output / 'summary.json', {'split': args.split, 'episodes': 40,
                      'seeds': [0, 1, 2], 'frames': 120, 'results': results})
            dump_json(status_path, {'state': 'step_complete', 'step': step,
                                    'completed': results})
        dump_json(status_path, {'state': 'complete', 'completed': results})
    except BaseException as exc:
        dump_json(status_path, {'state': 'failed', 'error': repr(exc), 'completed': results})
        raise


if __name__ == '__main__':
    main()
