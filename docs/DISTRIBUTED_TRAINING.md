# Multi-GPU training

For an intentional loss-weight change, `--new-objective` restores model, EMA,
optimizer and step into a new output run, permits the changed diffusion loss and
resets best criteria. Ordinary resume still rejects changed diffusion losses.
`--validate-at-start` logs a baseline and saves a recovery checkpoint before the
first update. Validation records are persisted even when the validation interval
is not divisible by the training log interval. Learning rate is logged with losses.

Optional W&B upload runs separately via `scripts/sync_diffusion_wandb.py --run RUN`.
It reads `RUN/wandb_spec.json` and the rank-zero `metrics.jsonl`, retaining the
local journal as the replay source. Only one uploader can hold its file lock;
resumes use the same W&B run ID and server-resumed history position. SDK failures
are retried without stopping DDP. Status distinguishes enqueued records from a
separate server verification; it does not promise uninterrupted network service.
Metric groups are `train`, `val` and `val_rollout`, with `/raw/<term>`,
`/weighted/<term>` and `/loss`; `trainer/step` is the common x axis. Network errors
cannot erase the local JSONL source. Install the optional `tracking` dependency
and configure W&B login before launching the uploader. Never put keys in configs.

Validation supports `train.val_sampling: stratified_action`: balance the full
global validation budget across action classes, select episodes in randomized
cycles and distinct windows within episodes, then shard the shuffled global list
across ranks. With 512 samples and 40 classes, each class contributes 12 or 13
windows. The remainder classes change with seed+9001+step; replaying the same step
reproduces the selection. Both teacher and generated-history evaluation use the
same selected windows. Rank zero persists the sample list and class counts.
Startup rejects train/val episode overlap or missing validation action classes.
Changing sampling protocol on resume resets best loss values; retain old best
assets in the old run and continue in a new run directory. Sampling noise means
single best scores should be supplemented with broader evaluation.
Changing global batch size in a stage transition also resets both best loss values,
because the validation sample count changes. The new run records its own best
checkpoints from `--validate-at-start` onward.

Fresh restart, 2026-09-26: denoiser starts at step zero on GPUs 0,1,2,3,
using the same frozen 50,000-step CVAE snapshot. Global batch 512 (128/rank),
learning rate 1e-4. Stage lengths 25k/25k/25k preserve total sample exposure
relative to batch128 with 100k/100k/100k stages, but do not preserve optimizer
trajectories. The existing GPUs 4–7 run is retained. GPUs 0/1/3 also have prior
experiments, so this new run shares their compute capacity.

Checkpoint policy: `last.pt` plus immutable `step_NNNNNN.pt` every 1,000 updates;
`best.pt` and `best.json` whenever EMA teacher-history validation improves;
`best_rollout.pt` and its metadata when EMA generated-history validation improves.
Best criteria are loss proxies, not full generative quality metrics. New runs
validate every 250 updates on 512 windows, including full generated-history
rollout. Original processes do not acquire these code changes dynamically.
Older overwritten best steps cannot be recovered from their scalar logs.

Tests: 21 automated tests passed; real two-GPU smoke produced and reloaded all
four checkpoint types. Full-rollout backward at local batch64 / 128 used about
4.67 / 8.80 GB peak allocated memory respectively on the tested GPU.

`moreact.train_distributed` uses PyTorch DDP with one process per GPU. The config
batch size is GLOBAL: batch 128 on four GPUs means 32 samples per GPU. Learning
rate, optimizer update count, EMA schedule and rollout curriculum remain unchanged.
The frozen CVAE and SMPL-X model are local replicas; only denoiser gradients sync.
The whole continuous-primitive loss is wrapped in one DDP forward. Contact losses
use global contact counts, including ranks with zero contacts, preserving the
original global-batch objective.

Example (physical GPU IDs remain visible; do not combine this mapping with a
different CUDA_VISIBLE_DEVICES ordering):

```bash
/data/users/autovla/.envs/remogen-motion-only/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 -m moreact.train_distributed \
  --config RUN/launch.yaml --output RUN --resume CHECKPOINT --devices 4,5,6,7
```

Rank zero writes logs and checkpoints. Checkpoints retain ordinary model keys,
optimizer/EMA/global step plus per-rank RNG states. Migration from single GPU
preserves learned state but changes sampling/RNG streams; it is not bitwise
equivalent to continuing the original single-GPU job. A shared global index list
is split into disjoint rank shards; validation covers the same global index range.

Verification: the 20 pre-existing tests passed; a new two-process Gloo test checks
contact loss and gradient equivalence with unequal contact counts and an empty
rank. Real two-GPU tests covered forward/backward, validation, checkpoint resume,
generated-history rollout and unchanged frozen VAE weights. Four GPUs completed
25 production-size global-batch-128 updates, validation and checkpoint save.

Benchmark: steady median 1.8605 s/update on GPU5 vs 0.7032 s/update on GPUs
2,5,6,7 (2.65x), FP32, all geometry losses, teacher history. This excludes warmup,
validation and checkpoint IO. GPUs 2,5,6,7 were used to avoid interrupting the live
GPU4 job; production migration targets the same-NUMA group 4,5,6,7. Rollout-stage
speedup can differ. Results are retained in the
[benchmark evidence](reports/20260926_artifact_cleanup/evidence/runs/ddp_benchmark/result.json).
Temporary benchmark checkpoints are covered by the
[cleanup record](reports/20260926_artifact_cleanup/README.md).

From-scratch run, 2026-09-26 16:57 UTC: the denoiser starts at step zero on GPUs
4,6,7 (three ranks, global batch 768, local 256), because GPU5 stays occupied by
the semantics captioner. Stage lengths 2000/17000/17000 preserve the 27.6M-sample
exposure of 3000/25000/25000 at batch 512, and the learning rate follows the same
per-sample decay from 1e-4 to zero instead of jumping at the batch change.
`diffusion_loss` adds `root_position: 30`; validation is `stratified_action` with
768 windows and `--validate-at-start` records the random-init baseline. The
frozen CVAE is the 50,000-step snapshot (sha256 `6093da86…591f`), whose 148
network tensors are bit-identical to the VAE embedded in the b512 checkpoints —
only the re-estimated `latent_scale` buffer differs. A two-step preflight at local
batch 256 with generated history forced peaked at 17.4 GB per rank (52,158 MiB
across three ranks). The GPU0–3 b512 run without root_position continues unchanged
as the control. W&B upload covers `train`, `val` and `val_rollout`; see the
[run report](reports/20260926_scratch_b768_root30/README.md).

`scripts/migrate_to_ddp.py` waits for an atomic completed checkpoint and available
GPUs, stops the old process temporarily, copies the checkpoint and starts DDP.
It terminates the old process only after a finite first DDP training step. On
startup failure it resumes the old process. A few updates after the checkpoint
can be replayed; checkpointed progress is preserved. State and errors are written
to `migration_status.json`. The original training lacks a live checkpoint signal,
so the initial switch waits for its scheduled 10,000-step checkpoint. New DDP
runs checkpoint every 1,000 steps.
