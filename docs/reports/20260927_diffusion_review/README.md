# Diffusion code review and fixes

2026-09-27. User requested history translation probability 25% and a review of
the diffusion stage. No production training process/run/checkpoint was modified.

## Findings and resolution

1. **P1 — inconsistent generated-history boundary targets.** Before this review,
   only noise-selected samples recomputed first-frame DTRANS/DJOINTS/DROT. With
   generated history but no augmentation selection (including rollout validation),
   position supervision used original GT while delta supervision still referenced
   the original GT predecessor. The regression probe reproduced a 0.54355 m maximum
   translation-delta inconsistency with the synthetic/untrained test model; this is
   a correctness witness, not a production error metric. The new geometry recipe
   enables `diffusion_loss_options.recompute_target_deltas: true`, consistently
   recomputing all samples' boundary deltas against actual supplied history before
   canonicalization and encoding. Future positions/poses and later deltas remain GT.
   Old configs retain old targets; normal resume rejects changes to these options.
   Both reusable audit scripts now honor this target convention.

2. **P2 — single-device rollout validation ignored.** `Trainer.validate()` ignored
   `train.validate_rollout`, although the DDP trainer honored it. The regression
   test observed missing `val_rollout_loss`. Single-device training now reports it,
   saves `best_rollout.pt/json`, restores the best value, and resets both best
   values on explicit new-objective or validation-sampling transitions. Validation
   preserves RNG and never enables history noise.

3. **P2 — single-device learning-rate restart ignored.** Single-device updates
   ignored `lr_schedule_start_step` while DDP used it. At step 1/origin 1/base 1e-4,
   the regression measured 9e-5 instead of 1e-4. Both trainers now use
   `learning_rate_at`; before the configured origin LR stays at the base rate,
   then decreases linearly to zero. Invalid origins are rejected at config loading.

The recipe probability is now **0.25**, with unchanged std=.02 m, max=.05 m,
FK weight20, root feature multiplier5, DM GT threshold1.5 m, contact threshold.10 m,
and no independent root_position loss. Stable docs updated; previous experiment
reports and run configs remain historical records.

## Review scope and remaining limitation

Inspected augmentation, normalization/canonicalization, target encoding, frozen
decoder autograd, x0 DDPM training/sampling, geometry and masked interaction
losses, generated-history feedback, EMA, validation, checkpoint/RNG resume, and
DDP mask normalization. Existing tests confirm actor future labels do not enter
the generator condition, VAE parameters stay frozen, root feature gradients scale
5x on only24 channels, and empty masks remain differentiable.

**Unresolved modeling limitation:** original GT future positions plus perturbed
history still reward immediate correction at the boundary. Recomputing deltas
removes inconsistent labels, not that preference. The frozen VAE's capability to
recover smoothly from drift and long-horizon quality require controlled training
and evaluation. No claim is made that these changes improve rollout drift.

## Evidence

Python: `/data/users/autovla/.envs/remogen-motion-only/bin/python`.
Working directory: `/data/autovla/projects/MoReAct`.

Before fixes:

```sh
python -m pytest tests/test_diffusion_review.py -q --basetemp=tmp/20260927_diffusion_review_before
```

Result: **3 failed in2.21s**, exit1, reproducing findings1–3 respectively.
Initial post-fix targeted run: **25 passed in15.99s**, exit0.

Final regression run:

```sh
python -m pytest tests/test_diffusion_review.py tests/test_history_augmentation.py tests/test_losses.py tests/test_training_generation.py tests/test_distributed.py tests/test_validation_sampling.py -q --basetemp=tmp/20260927_diffusion_review_final
```

Result: **30 passed in16.10s**, exit0. Includes generated-history delta consistency
in eval/no-grad mode (translation, joints and orientation), LR boundaries/config
validation, single-device rollout validation and RNG preservation, best checkpoint
resume/reset, legacy uninterrupted-vs-resumed equivalence, noisy geometry training,
frozen decoder gradient propagation, generation causality, 2-process CPU/gloo
masked gradient normalization, and stratified validation sampling.

Recipe loading and AST checks for changed audit scripts/DDP module also passed.
No full GPU/NCCL training smoke or quality evaluation was run. Test data and
checkpoints are synthetic and isolated; integration tests use local SMPL assets.
Cleanup inventory is recorded separately in `cleanup.json` after process checks.

All three isolated test directories were removed after archiving results and
checking for active pytest processes; formal runs and datasets were untouched.
