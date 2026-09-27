"""Migrate at a saved step; retain the old process for startup rollback."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moreact.config import dump_json
from moreact.train import load_checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--devices',default='4,5,6,7')
    parser.add_argument('--min-step',type=int,default=10000)
    args=parser.parse_args()
    source,output=Path(args.source).resolve(),Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    lock=(source/'ddp_migration.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status={'watcher_pid':os.getpid(),'source_pid':args.pid,'source':str(source),
            'output':str(output),'devices':args.devices,'minimum_checkpoint_step':args.min_step}
    def update(state,**kwargs):
        status.update(state=state,updated_unix=time.time(),**kwargs)
        dump_json(output/'migration_status.json',status)
    def alive():
        try:
            return str(source).encode() in (Path('/proc')/str(args.pid)/'cmdline').read_bytes()
        except FileNotFoundError:
            return False
    paused,switched,child=False,False,None
    try:
        update('waiting_for_checkpoint')
        while True:
            checkpoint=source/'last.pt'
            if checkpoint.exists():
                saved=load_checkpoint(checkpoint)
                if saved['step']>=args.min_step:
                    # Do not take over devices occupied by another job meanwhile.
                    usage=subprocess.check_output(['nvidia-smi',
                        '--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
                    memory={int(row.split(',')[0]):[int(x) for x in row.split(',')[1:]]
                            for row in usage.strip().splitlines()}
                    devices=[int(x) for x in args.devices.split(',')]
                    if memory[devices[0]][1]>=8192 and all(memory[x][0]<1024 for x in devices[1:]):
                        break
                    update('waiting_for_devices',checkpoint_step=saved['step'])
            if not alive():
                update('source_stopped_before_checkpoint')
                return
            time.sleep(10)
        if alive():
            os.kill(args.pid,signal.SIGSTOP);paused=True
        snapshot=output/'resume_source.pt'
        shutil.copy2(checkpoint,snapshot)
        saved=load_checkpoint(snapshot)
        update('starting_ddp',checkpoint_step=saved['step'])
        command=[sys.executable,'-m','torch.distributed.run','--standalone',
                 '--nproc_per_node='+str(len(args.devices.split(','))),
                 '-m','moreact.train_distributed','--config',str(output/'launch.yaml'),
                 '--output',str(output),'--resume',str(snapshot),'--devices',args.devices]
        with (output/'train.log').open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        update('starting_ddp',launcher_pid=child.pid,command=command)
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            if child.poll() is not None:
                raise RuntimeError('DDP startup exited: '+str(child.returncode))
            metrics=output/'metrics.jsonl'
            if metrics.exists():
                try:
                    record=json.loads(metrics.read_text().splitlines()[-1])
                except (IndexError,json.JSONDecodeError):
                    time.sleep(2);continue
                if record['step']>saved['step'] and math.isfinite(record['loss']):
                    switched=True
                    break
            time.sleep(2)
        if not switched:
            raise RuntimeError('DDP startup timed out')
        if paused and alive():
            os.kill(args.pid,signal.SIGTERM)
            os.kill(args.pid,signal.SIGCONT)
            paused=False
        update('running_ddp',first_step=record['step'])
        code=child.wait()
        update('complete' if code==0 else 'failed',exit_code=code)
    except Exception as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        update('failed',error=str(exc),source_will_resume=paused and not switched)
        raise
    finally:
        if paused and alive():
            os.kill(args.pid,signal.SIGCONT)


if __name__=='__main__':
    main()
