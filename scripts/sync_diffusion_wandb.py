"""Single-writer, resumable upload of durable diffusion JSONL metrics."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moreact.config import dump_json


def metric_payload(record):
    result = {'trainer/step': int(record['step'])}
    meta = {'grad_norm', 'learning_rate', 'rollout_probability', 'elapsed_seconds', 'world_size', 'baseline'}
    for key, value in record.items():
        if key == 'step':
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Nonfinite/non-numeric metric: '+key)
        if key in meta:
            result['trainer/'+key] = value
            continue
        prefix = 'train'
        if key.startswith('val_rollout_'):
            prefix, key = 'val_rollout', key[len('val_rollout_'):]
        elif key.startswith('val_'):
            prefix, key = 'val', key[len('val_'):]
        if key == 'loss':
            result[prefix+'/loss'] = value
        elif key.startswith('weighted_'):
            result[prefix+'/weighted/'+key[len('weighted_'):]] = value
        else:
            result[prefix+'/raw/'+key] = value
    return result


def complete_records(stream):
    """Never consume a partially flushed JSONL record."""
    while True:
        start = stream.tell()
        line = stream.readline()
        if not line or not line.endswith('\n'):
            stream.seek(start)
            return
        yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    root = Path(args.run).resolve()
    directory = root/'wandb_sync'
    directory.mkdir(exist_ok=True)
    lock = (directory/'sync.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import wandb
    spec = json.loads((root/'wandb_spec.json').read_text())
    status = {'pid': os.getpid(), 'id': spec['id'], 'entity': spec['entity'], 'project': spec['project']}
    def update(**values):
        status.update(values, updated_unix=time.time())
        dump_json(directory/'status.json', status)
    retries = 0
    while True:
        run = None
        try:
            config = json.loads((root/'config.json').read_text()) if (root/'config.json').exists() else spec['config']
            run = wandb.init(entity=spec['entity'], project=spec['project'], id=spec['id'],
                name=spec['name'], resume='allow', mode='online', dir=str(directory), config=config,
                settings=wandb.Settings(init_timeout=60, start_method='thread'))
            run.define_metric('trainer/step')
            run.define_metric('*', step_metric='trainer/step')
            for name in ('train/loss', 'val/loss', 'val_rollout/loss'):
                run.define_metric(name, step_metric='trainer/step', summary='min')
            # Server-resumed SDK history step, not a local cursor, is authoritative.
            last = int(run.step)-1 if run.resumed else -1
            update(state='online', url=run.url, last_enqueued_step=last, retries=retries)
            while not (root/'metrics.jsonl').exists():
                time.sleep(2)
            with (root/'metrics.jsonl').open() as stream:
                while True:
                    for record in complete_records(stream):
                        step = int(record['step'])
                        if step <= last:
                            continue
                        run.log(metric_payload(record), step=step)
                        last = step
                        update(state='online', last_enqueued_step=last)
                    launch = json.loads((root/'launch.json').read_text())
                    pid = launch['launcher_pid']
                    proc = Path('/proc')/str(pid)
                    try:
                        alive = str(root).encode() in (proc/'cmdline').read_bytes()
                    except FileNotFoundError:
                        alive = False
                    if not alive:
                        # Recheck EOF after process termination before finishing.
                        for record in complete_records(stream):
                            if record['step'] > last:
                                run.log(metric_payload(record), step=record['step']); last=record['step']
                        run.finish()
                        update(state='finished', last_enqueued_step=last)
                        return
                    time.sleep(2)
        except Exception as exc:
            retries += 1
            update(state='retrying', retries=retries, error=type(exc).__name__+': '+str(exc))
            print('W&B upload retry', retries, type(exc).__name__, flush=True)
            if run is not None:
                try:
                    run.finish(exit_code=1)
                except Exception:
                    pass
            time.sleep(min(30, 2**min(retries, 5)))


if __name__ == '__main__':
    main()
