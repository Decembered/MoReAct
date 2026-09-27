# History augmentation and root feature supervision

2026-09-27. Implementation and CPU checks only; no training run was launched,
resumed or modified. Existing runs/checkpoints retain their saved objectives.

## Source inspection

- DART: `/data/autovla/projects/DART/mld/train_mld.py`, `common_step` lines
  405–467 and curriculum lines 520–525: GT or generated history; Gaussian noise
  applied to future latent. No explicit history translation noise in this path.
- InterGen: `/data/users/autovla/reaction_baselines_20260820/intergen/models/nets.py`,
  `compute_loss`: motion diffusion and condition dropout. Dataset interhuman.py
  uses random crops/role changes; no explicit history translation corruption.
- InterGen `models/losses.py`, `forward_distance_map`: 1 m predicted-distance
  mask. MoReAct now offers a configurable GT-distance mask to prevent evasion.
  User selected 1.5 m for DM and retained 0.10 m for joint contact.

## Changes

- Geometry recipe: FK joint weight 20, root feature multiplier 5 over exactly 24
  channels with original B*F*276 mean denominator; root_position remains omitted.
- DM threshold 1.5 m using GT distances; DDP global selected-pair mean and empty
  mask differentiable zero reuse the tested masked_huber implementation.
- Diffusion history augmentation: p=.5, XY Gaussian std=.02 m, radial max=.05 m.
  Coherent shift per sample/primitive, before canonicalization, training only.
  Target future positions remain GT; first target deltas are recomputed for
  selected samples against actual corrupted history. No caller tensor mutation.
- Options/config validation, resume identity checks, audit-script propagation,
  docs and tests updated. CVAE and legacy configs keep prior behavior.

The corruption amplitudes are initial engineering settings, not tuned results.
The original GT position target still rewards immediate small-offset recovery;
boundary-delta consistency does not guarantee smooth recovery. Drift improvement
requires controlled training/evaluation, especially with the frozen CVAE.

## Validation

Command (project root, remogen-motion-only Python environment):

```sh
/data/users/autovla/.envs/remogen-motion-only/bin/python -m pytest tests/test_history_augmentation.py tests/test_losses.py tests/test_training_generation.py::test_diffusion_geometry_backpropagates_through_frozen_vae tests/test_distributed.py -q --basetemp=tmp/20260927_history_noise_tests
```

Result: **16 passed in 6.96s**, exit code 0. Checks include coherent bounded XY
shift and input immutability; selected boundary deltas; disabled RNG preservation;
invalid configs; exact root gradient multiplier; GT DM mask and empty-mask gradient;
legacy/new objective forward-backward through frozen VAE; noisy rollout feedback;
no augmentation during validation/eval; weighted-total accounting; resume rejection;
two-process CPU/gloo masked mean with an empty rank.

Temporary synthetic data/checkpoints were confined to
`tmp/20260927_history_noise_tests`; no formal datasets/checkpoints were used as
training inputs for these checks (SMPL body assets are used by integration tests).

After recording the passing results and checking for active pytest processes,
removed only the isolated test directory. File count and size are in `cleanup.json`.
Recipe loading and updated audit-script syntax checks also passed.
