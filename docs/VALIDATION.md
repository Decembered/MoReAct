# MoReAct implementation validation

## Shared frozen CVAE / RVQ understanding, 2026-09-26

Implementation, compatibility tests and bounded real-asset checks are recorded in
[the shared-RVQ report](reports/20260926_shared_rvq/README.md). Generation remains
continuous. The caption pipeline is implemented, but production semantic quality
has not been established and no long T5 training run was launched.

## Artifact retention update, 2026-09-26

Historical smoke/preflight/benchmark files below were archived for cleanup under
[the maintenance report](reports/20260926_artifact_cleanup/README.md).
Its manifest records the exact deletion status. Small configs, metrics and logs
are preserved in that report's `evidence/` tree using their original project paths.
Historical checkpoint/video paths in this document are provenance references,
not an indication that those files remain available. Production caches and live
training runs are retained. This maintenance pass did not rerun GPU experiments.

## Equal-role input preparation, 2026-09-24

- Automated suite: 18 passed, including explicit equal-role population statistics,
  stride-1 sampling and checkpoint statistics mismatch rejection.
- Full `data/interx_h2_f8` cache completed: 9,110 train, 570 val, 1,707 test
  episodes; one test episode (`G021T003A016R007`) skipped for insufficient length.
- Train-only shared normalization: actor and reactor each contribute 15,444,670
  window-frame observations (overlapping windows), weights 0.5/0.5; H=2, F=8,
  statistics stride=1. All means/stds finite; std floor 0.001.
- Four-primitive stride-1 datasets: 1,325,879 train, 82,081 val, 241,313 test
  windows. Four short train episodes are eligible for single-primitive statistics
  but not for the 34-frame training sequence.
- Split episode IDs disjoint; real normalized conditions load with finite values
  and shape [B,2,276]. Sampled reused arrays exactly match the source cache.
- Reports: `data/interx_h2_f8/preparation_report.json` and `input_validation.json`.
  Historical caches and running baseline training were left intact.

Date: 2026-09-23. Environment: Python 3.8, PyTorch 2.1.0, NVIDIA RTX 3090.

## Completed checks

| Check | Result |
| --- | --- |
| Automated tests | 15 passed: causal deltas, shared-frame invariance/round trip, CVAE history-boundary delta/BL/GT-contact FC losses, degenerate rotations, role overrides, split leakage, train-only normalization, overfitting, curriculum, VAE/diffusion exact CPU resume, frozen VAE, streaming causality, text-only CFG and metric units |
| Raw Inter-X preparation | 8 train + 8 val + 8 test episodes; 0 skipped; official splits retained |
| Cached geometry vs SMPL-X FK | Maximum absolute joint coordinate error `3.5763e-7 m` after Y-up → Z-up conversion, including shaped pelvis offsets |
| CVAE differentiable FK target parity | Two full-cache windows: mean joint error `9.01e-8 m`, maximum `3.69e-7 m` between transformed cached joints and SMPL-X FK |
| DART-aligned primitive smoke | `H=2,F=8` cache prepared for 8/8/8 train/val/test episodes with no skips; one CVAE train/validation step completed with all DART, BL and FC terms finite |
| Actual default CVAE size | Batch 128, 4 primitives, forward/backward + validation completed |
| Actual default diffusion size | Batch 128, 4 primitives, teacher-forcing step and full autoregressive forward/backward completed; peak allocated GPU memory about 4.81 GB; frozen VAE gradients: 0 |
| Small-model two-stage training | CVAE 200 steps + diffusion 200 steps, including all curriculum phases; finite losses |
| Real Inter-X fixed-batch overfit | 1,000 CVAE updates: reconstruction MSE `0.759252 → 0.002231` |
| Autoregressive generation | 64 frames generated from a 16-frame prefix, SMPL-X exported, MP4 rendered |
| Evaluation | VAE posterior reconstruction plus six diffusion condition settings over 2 validation episodes × 32 frames |

The real overfit check uses the first four training windows of the smoke cache,
smaller networks, learning rate 0.001, EMA 0.9 and a 1,000-step teacher-forcing
schedule. It demonstrates capacity to reconstruct the selected data, not
generalization. Its configuration is archived at
[the original overfit config](reports/20260926_artifact_cleanup/evidence/runs/real_overfit_vae/config.json).
The held-out reconstruction MSE remains around 1.259.

The full-size check uses production architecture and batch size with the smoke
cache; it verifies execution and memory feasibility, not full-data convergence.

## Artifacts

- `data/interx_smoke/preparation_report.json`: split counts, provenance, role status.
- `runs/smoke_vae/metrics.jsonl`, `runs/smoke_diffusion/metrics.jsonl`: short-run loss logs.
- `runs/real_overfit_vae/metrics.jsonl`: real-data overfit results.
- `outputs/smoke_vae_eval/report.json`: posterior reconstruction metrics.
- `outputs/smoke_eval/report.json`: text/actor condition comparisons and paired output sensitivity.
- `outputs/smoke_preview/preview.mp4`: 64-frame diagnostic video.
- `outputs/smoke_preview/{actor,reactor,target}_smplx.npz`: exported SMPL-X motions.

## Quality limits and ongoing training

The 200-step smoke weights are not converged. The diagnostic 64-frame rollout
has substantial drift (root ADE approximately 4.30 m, FDE approximately 7.69 m).
Its video is a pipeline check, not a successful reaction-generation result.
Condition ablations run correctly, but these short runs do not establish useful
actor dependence or text control.

The full-data pipeline was started on GPU 0, with default 300,000-step CVAE and
300,000-step diffusion curricula. At launch it was preparing raw Inter-X; full
training and its quality evaluation were not yet complete. Current status is
recorded in `runs/full/status.json`; read the per-stage logs and checkpoints for
actual progress. The initial launcher process ID was 561424; process IDs are
only valid while the corresponding process exists.

That running process loaded the earlier reconstruction-MSE plus `1e-4` KL
objective before the DART/BL/FC CVAE loss implementation was added. It remains
an earlier-objective baseline; its checkpoint is intentionally rejected by the
new resume compatibility check rather than mixing objectives mid-run.

The project default subsequently changed from the historical `H=16,F=8`
experiment above to the DART-aligned `H=2,F=8` primitive. New runs use
`data/interx_h2_f8`; historical smoke window configurations remain in the archived
evidence. The full-data baseline remains a separate retained run.

The role direction follows the user-selected Inter-X README convention plus
two motion-verified exceptions. Most roles remain annotation assumptions;
geometry/text screening candidates have not been promoted to ground truth.

## Reproduce

```bash
cd /data/autovla/projects/MoReAct
export MOREACT_PYTHON=/data/users/autovla/.envs/remogen-motion-only/bin/python
"$MOREACT_PYTHON" -m pytest -q
MOREACT_PYTHON="$MOREACT_PYTHON" bash scripts/smoke.sh
```

The current smoke recipe uses H=2 and current losses. It is not a bitwise
reproduction of the historical H=16 experiment. Repeating the historical overfit
requires restoring its matching implementation and rebuilding its small cache;
the archived config alone is not directly runnable against the current defaults.
