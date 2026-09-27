#!/usr/bin/env python3
"""Run the frozen-CVAE RVQ/caption experiment with durable stage status and logs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moreact.config import dump_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    run = ROOT / cfg['run']
    run.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    status = dict(pid=os.getpid(), config=cfg, state='running')
    def update(**kw):
        status.update(kw, updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        dump_json(run / 'status.json', status)
    def execute(stage, command):
        update(stage=stage, command=command)
        with (run / (stage+'.log')).open('a') as stream:
            proc = subprocess.Popen([sys.executable, '-u']+command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            update(child_pid=proc.pid)
            code = proc.wait()
        if code:
            raise RuntimeError('%s failed with exit %d; see %s' % (stage, code, run / (stage+'.log')))
        update(child_pid=None)
    def gpu_wait():
        update(stage='waiting_for_gpu_memory')
        index = cfg['device'].split(':')[-1]
        while True:
            result = subprocess.check_output(['nvidia-smi', '--id='+index, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
            free = int(result.strip())
            update(gpu_free_mib=free)
            if free >= cfg['min_free_mib']:
                return
            time.sleep(30)
    prefix = ['-m', 'moreact', 'semantics']
    data = cfg['data']
    try:
        update(stage='waiting_for_latents')
        while not (ROOT / data / 'latents/manifest.json').exists():
            pid = cfg['cache_pid']
            try:
                state = Path('/proc/%d/stat' % pid).read_text().split()[2]
                if state == 'Z':
                    raise RuntimeError('Latent extraction exited without manifest')
            except FileNotFoundError:
                raise RuntimeError('Latent extraction stopped without manifest')
            update(latent_files=len(list((ROOT / data / 'latents').glob('*.npz'))))
            time.sleep(15)
        execute('fit_rvq', prefix+['fit-rvq', '--cache', data+'/latents', '--output', cfg['run']+'/rvq',
                '--epochs', str(cfg['rvq_epochs']), '--seed', '42', '--device', 'cpu'])
        execute('cache_tokens', prefix+['cache-tokens', '--cache', data+'/latents', '--rvq', cfg['run']+'/rvq/best.pt',
                '--output', data+'/tokens', '--device', 'cpu'])
        gpu_wait()
        execute('language_preflight', ['scripts/semantic_language_preflight.py', '--cache', data+'/tokens',
                '--output', cfg['run']+'/language_preflight.json', '--device', cfg['device']])
        train = prefix+['train', '--cache', data+'/tokens', '--output', cfg['run']+'/captioner', '--base-model', cfg['base_model'],
                       '--device', cfg['device'], '--batch-size', '1', '--gradient-accumulation', '16',
                       '--gradient-checkpointing', '--learning-rate', '0.0001', '--validate-every', '100', '--seed', '42']
        execute('train_initial', train+['--steps', '100'])
        comparison = cfg['comparison']
        # Baseline evaluation has its own process and can finish while cache/RVQ train.
        execute('compare_initial', ['scripts/compare_bridge_semantics.py', '--mode', 'moreact', '--output', comparison,
                '--cache', data+'/tokens', '--checkpoint', cfg['run']+'/captioner/best.pt', '--device', cfg['device']])
        if (ROOT / comparison / 'bridge.json').exists():
            execute('summarize_initial', ['scripts/compare_bridge_semantics.py', '--mode', 'summarize', '--output', comparison])
        execute('train_continue', train+['--steps', str(cfg['steps']), '--resume', cfg['run']+'/captioner/last.pt'])
        final = ROOT / (comparison+'_final')
        final.mkdir(exist_ok=False)
        import shutil
        shutil.copy2(ROOT / comparison / 'selection.json', final / 'selection.json')
        if (ROOT / comparison / 'bridge.json').exists():
            shutil.copy2(ROOT / comparison / 'bridge.json', final / 'bridge.json')
        execute('compare_final', ['scripts/compare_bridge_semantics.py', '--mode', 'moreact', '--output', str(final),
                '--cache', data+'/tokens', '--checkpoint', cfg['run']+'/captioner/best.pt', '--device', cfg['device']])
        if (final / 'bridge.json').exists():
            execute('summarize_final', ['scripts/compare_bridge_semantics.py', '--mode', 'summarize', '--output', str(final)])
        update(state='complete', stage='complete')
    except Exception as exc:
        update(state='failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
