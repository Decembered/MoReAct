"""Synchronous data parallel training with a fixed global batch size.

Launch with torchrun; --devices maps local ranks to physical visible CUDA IDs.
The whole primitive chain is one DDP forward, including decoded supervision.
"""
import argparse
from contextlib import contextmanager
from datetime import timedelta
import json
import os
import shutil
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import load_config, dump_json
from .losses import GLOBAL_MASK_NORMALIZATION
from .validation import validation_indices
from .train import (Trainer, batch_at, load_checkpoint, save_checkpoint, rng_state,
                    restore_rng, seed_all, preserved_rng, rollout_probability, learning_rate_at)


@contextmanager
def global_contact_means():
    token = GLOBAL_MASK_NORMALIZATION.set(True)
    try:
        yield
    finally:
        GLOBAL_MASK_NORMALIZATION.reset(token)


class LossForward(torch.nn.Module):
    def __init__(self, trainer):
        super().__init__()
        self.network = trainer.model
        self.trainer = trainer

    def forward(self, batch, probability):
        with global_contact_means():
            return self.trainer.losses(batch, self.network, probability)


class DistributedTrainer(Trainer):
    def __init__(self, cfg, kind, output, resume=None, vae_path=None, stage2=False, new_objective=False):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        if cfg['train']['batch_size'] % self.world:
            raise ValueError('Global batch must divide evenly across ranks')
        self.local_batch = cfg['train']['batch_size'] // self.world
        root = Path(output)
        super().__init__(cfg, kind, root / '.ranks' / str(self.rank),
                         vae_path=vae_path, resume=resume, new_objective=new_objective,
                         resume_overrides=('batch_size','stage1_steps','stage2_steps','stage3_steps',
                            'learning_rate','lr_schedule_start_step','validate_rollout') if stage2 else ())
        self.best_val = float('inf')
        self.best_rollout = float('inf')
        if resume:
            saved_best = load_checkpoint(resume)
            self.best_val = saved_best.get('best_val', float('inf'))
            self.best_rollout = saved_best.get('best_rollout', float('inf'))
            # A changed global validation batch changes the score's sampling
            # distribution, so old best values cannot select this run's weights.
            if (new_objective or
                cfg['train'].get('val_sampling', 'sequential') != saved_best['config']['train'].get('val_sampling', 'sequential') or
                cfg['train']['batch_size'] != saved_best['config']['train']['batch_size']):
                self.best_val = self.best_rollout = float('inf')
        self.output = root
        self.ddp = DDP(LossForward(self), device_ids=[self.device.index],
                       output_device=self.device.index, broadcast_buffers=False)
        if resume:
            saved = load_checkpoint(resume)
            if saved.get('world_size') == self.world and 'distributed_rng' in saved:
                restore_rng(saved['distributed_rng'][self.rank])
            else:
                seed_all(cfg['train']['seed'] + self.step * self.world + self.rank)
        else:
            seed_all(cfg['train']['seed'] + self.rank)
        if self.rank == 0:
            dump_json(root / 'config.json', cfg)
            dump_json(root / 'distributed.json', {'world_size': self.world,
                'global_batch_size': cfg['train']['batch_size'], 'local_batch_size': self.local_batch,
                'resumed_from': str(resume), 'start_step': self.step,
                'note': 'Rank RNG changes when migrating from single GPU; optimizer/EMA/step preserved.'})

    def averaged(self, terms):
        keys = sorted(terms)
        values = torch.stack([torch.as_tensor(terms[k], device=self.device).detach() for k in keys])
        dist.all_reduce(values)
        values /= self.world
        return {k: float(v) for k, v in zip(keys, values)}

    @torch.no_grad()
    def validate(self):
        results = []
        rollout_results = []
        selection = validation_indices(self)
        with preserved_rng(), global_contact_means():
            seed_all(self.cfg['train']['seed'] + 9001 + self.rank +
                     (self.step if self.cfg['train'].get('val_sampling') == 'stratified_action' else 0))
            for i in range(self.cfg['train']['val_batches']):
                start = i * self.cfg['train']['batch_size'] + self.rank * self.local_batch
                indices = selection[start:start + self.local_batch]
                batch = batch_at(self.validation, indices, self.device)
                results.append(self.losses(batch, self.ema))
                if self.cfg['train'].get('validate_rollout', False):
                    rollout_results.append(self.losses(batch, self.ema, 1.))
        means = {k: torch.stack([x[k] for x in results]).mean() for k in results[0]}
        output = {'val_' + k: v for k, v in self.averaged(means).items()}
        if rollout_results:
            means = {k: torch.stack([x[k] for x in rollout_results]).mean() for k in rollout_results[0]}
            output.update({'val_rollout_' + k: v for k,v in self.averaged(means).items()})
        return output

    def checkpoint(self, name='last.pt'):
        states = [None] * self.world
        dist.all_gather_object(states, rng_state())
        if self.rank == 0:
            state = {'format_version': 1, 'kind': self.kind, 'step': self.step,
                'config': self.cfg, 'model': self.model.state_dict(), 'ema': self.ema.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'rng': states[0],
                'distributed_rng': states, 'world_size': self.world,
                'best_val': self.best_val, 'best_rollout': self.best_rollout,
                'stats': {k:v.cpu() for k,v in self.stats.items()},
                'data_digest': self.dataset.manifest['digest'], 'overfit': False}
            if self.kind == 'diffusion':
                state['vae'] = self.vae.state_dict()
            save_checkpoint(self.output / name, state)
            if name == 'last.pt':
                archive = self.output / ('step_%06d.pt' % self.step)
                if not archive.exists():
                    # Atomic last.pt replacement makes an immutable hard link safe.
                    try:
                        os.link(self.output / name, archive)
                    except OSError:
                        shutil.copy2(self.output / name, archive)
        dist.barrier()

    def run(self, steps=None, rollout_override=None, validate_at_start=False):
        cfg = self.cfg['train']
        total = sum(cfg[k] for k in ('stage1_steps','stage2_steps','stage3_steps'))
        stop = total if steps is None else steps
        if not self.step < stop <= total:
            raise ValueError('Invalid target step')
        first, started = self.step, time.perf_counter()
        self.model.train()
        log = (self.output / 'metrics.jsonl').open('a') if self.rank == 0 else None
        try:
            if validate_at_start:
                baseline = {'step': self.step, **self.validate(), 'baseline': True}
                teacher_best = baseline['val_loss'] < self.best_val
                rollout_best = baseline.get('val_rollout_loss', float('inf')) < self.best_rollout
                self.best_val = min(self.best_val, baseline['val_loss'])
                self.best_rollout = min(self.best_rollout, baseline.get('val_rollout_loss', float('inf')))
                if self.rank == 0:
                    log.write(json.dumps(baseline, allow_nan=False)+'\n'); log.flush()
                self.checkpoint()
                for improved, filename, key in ((teacher_best, 'best', 'val_loss'),
                                                (rollout_best, 'best_rollout', 'val_rollout_loss')):
                    if improved:
                        self.checkpoint(filename+'.pt')
                        if self.rank == 0:
                            dump_json(self.output/(filename+'.json'), {'step': self.step, key: baseline[key]})
            while self.step < stop:
                # Identical global sampling list, disjoint shards across ranks.
                generator = torch.Generator().manual_seed(self.cfg['train']['seed'] + self.step)
                ids = torch.randint(len(self.dataset), (cfg['batch_size'],), generator=generator)
                ids = ids[self.rank*self.local_batch:(self.rank+1)*self.local_batch]
                batch = batch_at(self.dataset, ids, self.device)
                probability = rollout_probability(self.step, cfg) if rollout_override is None else rollout_override
                for group in self.optimizer.param_groups:
                    group['lr'] = learning_rate_at(self.step, cfg)
                self.optimizer.zero_grad(set_to_none=True)
                terms = self.ddp(batch, probability)
                finite = torch.isfinite(terms['loss']).to(torch.int)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError('Nonfinite distributed loss')
                terms['loss'].backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
                self.optimizer.step()
                with torch.no_grad():
                    for dst,src in zip(self.ema.parameters(),self.model.parameters()):
                        dst.lerp_(src,1-cfg['ema'])
                    for dst,src in zip(self.ema.buffers(),self.model.buffers()):
                        dst.copy_(src)
                self.step += 1
                record = {'step':self.step,'rollout_probability':probability, 'learning_rate': self.optimizer.param_groups[0]['lr'],
                          **self.averaged(terms), 'grad_norm':float(norm)}
                if self.step % cfg['val_interval'] == 0 or self.step == stop:
                    record.update(self.validate())
                    if record['val_loss'] < self.best_val:
                        self.best_val = record['val_loss']
                        self.checkpoint('best.pt')
                        if self.rank == 0:
                            dump_json(self.output/'best.json',{'step':self.step,'val_loss':self.best_val})
                    if record.get('val_rollout_loss',float('inf')) < self.best_rollout:
                        self.best_rollout = record['val_rollout_loss']
                        self.checkpoint('best_rollout.pt')
                        if self.rank == 0:
                            dump_json(self.output/'best_rollout.json',{'step':self.step,'val_rollout_loss':self.best_rollout})
                if self.step % cfg['log_interval'] == 0 or self.step in (first+1,stop) or 'val_loss' in record:
                    record['elapsed_seconds'] = time.perf_counter()-started
                    record['world_size'] = self.world
                    if self.rank == 0:
                        log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
                        print(json.dumps(record),flush=True)
                if self.step % cfg['save_interval'] == 0 or self.step == stop:
                    self.checkpoint()
        finally:
            if log is not None:
                log.close()
        return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume')
    parser.add_argument('--vae')
    parser.add_argument('--kind',choices=['vae','diffusion'],default='diffusion')
    parser.add_argument('--devices',default='4,5,6,7')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--rollout-override',type=float, help='Smoke testing only')
    parser.add_argument('--stage2',action='store_true',help='Explicit curriculum/batch transition into a new run')
    parser.add_argument('--new-objective',action='store_true',help='Explicit changed-loss continuation into a new run')
    parser.add_argument('--validate-at-start',action='store_true')
    args=parser.parse_args()
    devices=[int(x) for x in args.devices.split(',')]
    if len(devices)!=int(os.environ['WORLD_SIZE']) or len(set(devices))!=len(devices):
        raise ValueError('Devices must uniquely match WORLD_SIZE')
    device=devices[int(os.environ['LOCAL_RANK'])]
    torch.cuda.set_device(device)
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    try:
        cfg=load_config(args.config);cfg['train']['device']='cuda:'+str(device)
        cfg['train']['threads']=1
        if args.new_objective and (not args.resume or (Path(args.output)/'metrics.jsonl').exists()):
            raise ValueError('New objective requires resume and a new output run')
        if args.stage2:
            if not args.resume or (Path(args.output)/'metrics.jsonl').exists():
                raise ValueError('Stage2 transition requires a source checkpoint and a new output run')
            source = load_checkpoint(args.resume)
            if cfg['train']['stage1_steps'] != source['step'] or cfg['train'].get('lr_schedule_start_step') != source['step']:
                raise ValueError('Stage2 and LR schedule must start at checkpoint step')
        DistributedTrainer(cfg,args.kind,args.output,args.resume,args.vae,args.stage2,args.new_objective).run(
            args.steps,args.rollout_override,args.validate_at_start)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
