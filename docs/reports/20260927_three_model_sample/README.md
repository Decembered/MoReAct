# Three-model sampled rollout evaluation

2026-09-27. Completed 120 rollouts: 40 paired episodes x 3 models on physical GPU 5.
No training processes, checkpoints, or training configuration were changed.

## Protocol

- GPU 4/6/7: diffusion_scratch_b768_root30_20260926, step_008500.pt, EMA, guidance 1.
- GPU 0/1/2/3: diffusion_stage2_b512_stratified_20260926, step_021000.pt, EMA, guidance 1.
- ReMoGen: released interx_hhi_adapter/checkpoint.pt and mvae_hml3d/checkpoint.pt;
  native HHI rollout, guidance 5, predicted-joint feedback, no FWSR or floor correction.
- Select 20 action categories with seed 20260927, one random eligible episode per
  category in each of the train/test splits: 20 train + 20 test, no overlap.
  Eligibility requires at least 123 prepared frames and a matching official-format
  source with the same actor/reactor roles. This excludes short episodes and only
  covers half of the 40 action classes; it is not a full-dataset benchmark.
- Same source indices, roles, first caption, H=2, F=8, and 120 future frames. No
  best-of-N selection or target reactor feedback after initialization.
- Seed 0: MoReAct resets its generator per episode; official native rollout seeds
  once per run. Noise tensors are not identical between different implementations.
- Verified all 123 actor body-pose rotation matrices agree to <1e-4 across the two
  source representations. Verified native output captions, counts, finite values,
  and exact 120 x 22 x 3 shapes before aggregation.
- Score each model against its corresponding native ground-truth body. MoReAct
  uses gender/betas and FK repair; official ReMoGen uses neutral/zero betas and its
  native predicted-joint pipeline. Native GT rigid-alignment residual averaged
  3.15 cm (maximum 5.69 cm), so cross-pipeline differences are not purely model quality.
- The existing source FPS discrepancy remains unresolved. Comparisons use source
  frame indices; no physical duration, speed, or jerk claims are made.
- Train/test labels follow MoReAct's official Inter-X split manifest. We did not
  independently audit the released ReMoGen checkpoint's original training membership.

## Results

Episode-macro means, cm, lower is better. Joint error uses 22 joints. Root-aligned
joint error subtracts each body's pelvis translation, without rotation alignment.
Pair error is absolute error in actor/reactor pelvis distance.

| Split | Model | World joint error | Root ADE | Root FDE | Root-aligned joint error | Pair error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train | GPU 4/6/7, 8500 | 37.44 | 34.31 | 62.83 | 11.38 | 19.28 |
| Train | GPU 0-3, 21000 | 27.10 | 22.48 | 33.83 | 12.57 | 12.83 |
| Train | ReMoGen | 51.50 | 49.02 | 72.08 | 13.74 | 25.34 |
| Test | GPU 4/6/7, 8500 | 37.92 | 36.79 | 70.81 | 11.38 | 19.17 |
| Test | GPU 0-3, 21000 | 30.59 | 28.32 | 52.19 | 12.82 | 16.61 |
| Test | ReMoGen | 56.40 | 53.97 | 81.73 | 13.72 | 32.60 |

GPU 4/6/7 has the lowest root-aligned pose error, while GPU 0-3 has lower mean
trajectory/world errors. The scratch model's test root ADE rises from 11.27 cm
in the first 40 frames to 61.58 cm in the last 40; GPU 0-3 rises from 10.15 to
45.39 cm. Accumulated trajectory drift is the main observed weakness. The two
checkpoints have different training exposure and recipes, so these results do
not isolate the effect of the root-position loss or training from scratch.

Paired episode bootstrap, 10,000 resamples, seed 20260927: on test, GPU 4/6/7
minus GPU 0-3 world joint error is +7.33 cm (95% interval [-1.07, 19.92]); it wins
11/20 episodes despite a worse mean. Against ReMoGen its mean difference is
-18.48 cm ([-38.82, 3.26]), winning 15/20. These intervals cross zero. Test pose
differences are -1.44 cm ([-2.85, -0.11]) vs GPU 0-3 and -2.34 cm ([-4.03, -0.53])
vs ReMoGen. Bootstrap intervals reflect only this episode sample, not seed variance,
and do not correct for multiple metrics or native-body differences.

These are reference-motion errors, not semantic success, contact accuracy, or
complete generative-quality judgments. One stochastic seed cannot characterize
generation diversity. No mesh penetration or visual-quality rating was performed.

## Artifacts and Reproduction

Formal artifacts are retained at `outputs/three_model_sample_20260927/`:
selection/protocol/checkpoint hashes, per-episode metrics with first/last 40-frame
breakdowns, aggregate results, paired bootstrap intervals, geometry residuals,
comparison.png, both MoReAct joint trajectories, official native/per-episode PKLs,
official input and inference logs. No temporary checkpoint copies were created.
The script in this report is retained as reproducibility evidence.

From `/data/autovla/projects`, using the remogen-motion-only environment:

```bash
PY=/data/users/autovla/.envs/remogen-motion-only/bin/python
SCRIPT=MoReAct/docs/reports/20260927_three_model_sample/evaluate.py
$PY "$SCRIPT" select
CUDA_VISIBLE_DEVICES=5 $PY "$SCRIPT" ours
CUDA_VISIBLE_DEVICES=5 $PY "$SCRIPT" remogen
$PY "$SCRIPT" summarize
```

Generation stages reuse their existing outputs. For a different checkpoint or
selection, use a new output directory rather than mixing cached artifacts.
All stages exited successfully. Evidence JSON files are copied beside this report.
