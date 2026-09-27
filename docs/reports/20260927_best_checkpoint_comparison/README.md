# Continued-training checkpoint comparison

2026-09-27. Current continued-training lineage is diffusion_stage2_b512_stratified_20260926
(GPUs0–3), about21950 updates at inspection. Snapshot best.pt=13250 and best_rollout.pt=19750.
Compare both to previous8750 snapshot, not to the separate scratch/root30 experiment.
Formal immutable copies and artifacts retained in outputs/best_compare_20260927 for user
review; no training changes. Scripts moved from completed tmp task to evidence, empty tmp
removed. Generation/render/evaluation completed successfully on GPU5 / CPU.

## Protocol

Two previous hug/handshake test episodes, same first2 history frames, seed0, guidance1,
text, actor observations, full192/133 future frames, generated reactor feedback.
Visual panels:8750 / best13250 / rollout19750 / truth; identical coordinates and bounds.
Source frames synchronized, FPS discrepancy remains unresolved; no physical speed claims.

Broader check: reuse all20 test episodes from outputs/three_model_sample_20260927/selection.json,
one episode each from20 previously selected action classes, 120 future frames. Re-run ALL
three MoReAct checkpoints with current inference code, cached identical caption embeddings,
per-episode RNG seed0. Exclude initialization; episode-macro metrics. Reuse corresponding
ReMoGen outputs from previous evaluation (official native guidance5, different neutral body
and forward-difference conditioning). Cross-ReMoGen comparison is pipeline-specific; intra-
MoReAct comparison uses identical geometry and conditions. One seed and20 episodes only.

| Model | World joint cm | Root ADE cm | Root-aligned joint cm | Pair distance error cm |
|---|---:|---:|---:|---:|
| old8750 |31.59|29.88|10.79|16.85|
| best13250 |30.31|28.55|10.89|15.98|
| rollout19750 |30.23|28.38|12.84|16.07|
| ReMoGen |56.40|53.97|13.72|32.60|

best13250 world joint improvement4.06%, wins12/20; paired episode-bootstrap95%CI for
new-minus-old is[-5.31,+2.53]cm, so a stable overall improvement is not established.
rollout19750 wins8/20 on world joints; root-aligned pose degrades2.05cm, CI[0.87,3.39].
Validation now redraws stratified samples each checkpoint, so best metadata is not a
fixed-set global optimum and old/new validation losses should not be directly compared.

Full-clip hug world joint error28.99→22.50/22.58cm; handshake31.17→16.59/17.96cm.
However hug maximal adjacent-frame pelvis displacement2.33→19.28/28.68cm, with large
new jumps at primitive boundaries (source frames162/66). Thus mean reference errors
improve while temporal stability regresses in this example. Cause not isolated here.
No claim that all added motion is jitter: contact/trajectory dynamics need separate audit.

## Reproduce

Run evidence/generate.py, evaluate.py, render.py using the remogen-motion-only environment
from MoReAct root. Generation snapshots CURRENT live best files, so future reruns may
select later checkpoints. Saved snapshots in outputs/best_compare_20260927 retain exact
weights used here. evaluate.py uses saved snapshots and old8750; render uses saved motions.
