# MoReAct / official ReMoGen paired generation preview

2026-09-26. Two previously shown test clips, seed 0; no best-of-N selection.
MoReAct uses outputs/stage2_best_gifs/checkpoint.pt (step 8750, EMA), guidance 1.
ReMoGen uses remogen_official_release/checkpoints/interx_hhi_adapter/checkpoint.pt,
its released VAE, native HHI rollout without FWSR, guidance 5, predicted-joint feedback.
Both use H=2, F=8, the same interaction text and semantic role direction; reactor history
is generated after initialization. Official forward differences retain one-frame lookahead;
MoReAct uses backward differences. This is a deployed-pipeline comparison, not a causal
architecture-controlled benchmark. Two clips / one seed cannot establish overall superiority.

Artifacts: /data/autovla/projects/MoReAct/outputs/remogen_generation_comparison
contains official input, native eval.pkl, per-clip PKLs, aligned skeleton NPZs, GIFs, MP4s,
metrics and logs. Retained for visual review and reproducibility. No training jobs modified.

Reproduce using /data/users/autovla/.envs/remogen-motion-only/bin/python:
run evidence/generate.py from /data/autovla/projects/remogen_official_release, then
evidence/render.py from /data/autovla/projects/MoReAct (use absolute script paths).
Generator reuses existing native outputs if present.

## Comparison protocol and limitations

Metrics exclude two initialization frames and use the common available future frames
(191 hug, 132 handshake). Each model is scored against its own corresponding GT body.
ReMoGen is neutral/zero betas; MoReAct retains gender/betas. For visual alignment only,
a single rigid transform fits both GT bodies over the clip; predictions are never used
for fitting. No scale adjustment. GT residual is 6.42 cm / 5.44 cm, reflecting incompatible
body preprocessing and grounding. The reference panel shows MoReAct GT; ReMoGen's own
GT is retained in aligned NPZ. Metrics are thus indicative, not strictly identical-geometry.

Hug: joint error 28.92 vs 60.31 cm (MoReAct vs ReMoGen); root 28.67 vs 57.65 cm.
Handshake: joint error 31.18 vs 25.62 cm; root 29.77 vs 24.68 cm.
Root-aligned joint errors: hug 11.96 vs 14.38 cm, handshake 6.08 vs 7.28 cm.
Both remain imperfect; relative placement/contact is still problematic.

## FPS discrepancy requiring follow-up

MoReAct config assumes raw120 / target30 and samples stride4. The bridge reference
labels the same sampled poses as10 FPS. Both clips' actor body rotation matrices match
at every sampled index (max abs error 1.79e-7), so these are NOT separately resampled30/10
trajectories. Raw NPZ stores no FPS field. Existing bridge raw-conversion provenance
assumes source40/target10, with local conversion evidence. This comparison does not settle
physical source FPS. GIFs synchronize source indices at30 samples/sec, explicitly labeled;
physical timing and velocity-based metrics are not compared. Previous6.4s/4.4s durations
are configured playback durations, not independently verified capture durations.
Audit authoritative release timing before claiming true30FPS generation or interpreting
velocity/contact thresholds in seconds. No cache or training changes made in this task.

Temporary scripts were moved here as lightweight reproducibility evidence; empty task
directory removed. Formal outputs retained as requested visual artifacts.
