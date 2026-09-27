"""Wait for a completed CVAE checkpoint, then evaluate held-out episodes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moreact.config import dump_json
from moreact.train import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args()
    run = Path(args.run).resolve()
    lock = (run / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = json.loads((run / 'config.json').read_text())
    target = sum(cfg['train'][k] for k in ('stage1_steps', 'stage2_steps', 'stage3_steps'))
    status = {'watcher_pid': os.getpid(), 'training_pid': args.pid,
              'target_steps': target, 'device': args.device, 'frames_per_episode': 120}
    def update(state, **kwargs):
        status.update(state=state, updated_unix=time.time(), **kwargs)
        dump_json(run / 'evaluation_status.json', status)
    update('waiting_for_training')
    try:
        while True:
            command = Path('/proc') / str(args.pid) / 'cmdline'
            alive = command.exists() and str(run).encode() in command.read_bytes()
            lines = (run / 'metrics.jsonl').read_text().splitlines()
            try:
                step = json.loads(lines[-1])['step'] if lines else 0
            except json.JSONDecodeError:
                time.sleep(30)
                continue
            if step >= target or not alive:
                saved = load_checkpoint(run / 'last.pt')
                if saved['step'] >= target:
                    break
                if not alive:
                    update('training_stopped_before_completion', checkpoint_step=saved['step'])
                    return
            time.sleep(30)
        for split in ('val', 'test'):
            output = run / ('final_eval_' + split)
            update('evaluating', split=split, checkpoint_step=saved['step'])
            with (run / ('final_eval_' + split + '.log')).open('a') as log:
                subprocess.run([sys.executable, '-u', '-m', 'moreact', 'evaluate',
                                '--checkpoint', str(run / 'last.pt'), '--output', str(output),
                                '--split', split, '--limit', '0', '--frames', '120',
                                '--device', args.device], cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        update('complete')
    except Exception as exc:
        update('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
