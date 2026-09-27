# Drift supervision and feedback causality diagnosis

2026-09-27. All experiments completed on physical GPU 5. Training processes and
checkpoint parameters were not modified. Latent-only optimization below modifies
an isolated input tensor, never model weights or a training optimizer.

## Conclusion

The current root target penalizes all existing drift immediately. The current
joint-velocity boundary target also encourages returning to the original GT,
rather than preserving true GT velocity. This is a confirmed supervision defect
for smooth recovery from perturbed histories. Root-only latent optimization can
indeed lower root error by producing a large first-frame jump.

However, the frozen diffusion sampler does NOT usually execute this aggressive
correction in the controlled experiments: it retains almost all injected root
offset and heading. Existing generated-history jumps largely disappear when
local history pose/dynamics are replaced, even while endpoint position and yaw
drift are held fixed. Therefore the proposed complete chain
"root30 -> aggressive catch-up -> observed worse rollout" is not established.
The stronger supported explanation is weak recovery from history drift plus
local-history feedback instability, with a supervision target that would also
reward an undesirable recovery strategy if optimized aggressively.

## Setup

- Fixed checkpoints: root30 = scratch_b768 step8500; baseline = stage2_b512
  step21000. Same files as the preceding three-model comparison; hashes copied
  in checkpoints.json. These differ in training exposure/recipe and are not a
  causal ablation of root weight.
- Twelve episodes: first six train and six test from the previous deterministic
  selection, spanning six action categories. Selection fixed before results.
- EMA models, frozen VAE, guidance1, H2/F8, same actor/text/GT/body assets.
- Three seeds 0/1/2 for generation. Full-rollout segments use seed*1000+cutoff;
  intervention arms use identical noise at the same cutoff. This differs from
  the earlier 40-episode evaluation's single continuous RNG stream.
- Injection window ends at frame42; actual-history interventions at frames42/82.
- Metres/source-frame internally; reported cm and degrees. Physical FPS remains
  unresolved, so no m/s or m/s^2 claims.
- 720 injection results; 216 latent-gradient probes; 72 full 120-frame rollouts;
  720 actual-history intervention results; 24 analytic target witnesses;
  168 decoder-fit snapshots; 288 local-state intervention results.

## 1. Target construction: verified numerically

With generated previous position p_hat0 = p_GT0 + e, matching the original GT
future implies first displacement delta_p_GT1 - e. Root MSE penalizes residual
drift at every future frame, including the first, not only an 8-frame endpoint.
Both prediction and target joint-velocity sequences prepend the same generated
history. Thus their first velocity difference is p_hat1 - p_GT1: history cancels.

For a coherent +30cm translated history, using valid GT future features gives:

| Candidate future | root MSE x30 | current joint velocity x100 | true-GT velocity x100 |
| --- | ---: | ---: | ---: |
| Snap directly to original GT | 0 | 0 | 0.1875 |
| Preserve offset and GT motion increments | 0.9000 | 0.1875 | approximately 0 |

This exact witness passed on all 24 model/episode cases. It concerns these loss
components, not a claim that the entire learned objective always prefers snapping.
Root angular velocity uses the analogous generated-history target construction.

There is also a conflicting feature-target contract at the boundary: transformed
GT TRANSL/JOINTS remain at original GT positions, while stored DTRANS/DJOINTS retain
increments from the ORIGINAL GT previous frame. They are not recomputed against
generated history. On all12 episodes, the clean root-delta consistency residual
was <=5.97e-8m; translating history by30cm creates exactly30cm inconsistency in
both root and joint delta targets. Feature reconstruction therefore supervises
position and delta channels that cannot jointly describe a consistent transition
from the supplied history. The frozen VAE was trained with explicit delta
consistency. This is a plausible contributor to its drifted-history reconstruction
failure, but its individual causal contribution was not isolated by retraining.

## 2. Gradient directions: current velocity agrees with root

Use the actual encode/corrupt/denoise/decode path at diffusion timesteps0/5/9,
then differentiate each term w.r.t. predicted latent, with frozen EMA/decoder.
These are output-latent gradients, not full optimizer parameter gradients.
The baseline root30 term is evaluated diagnostically, although not in its recipe.

Root30 checkpoint, +/-30cm injected translation, 72 probes:

| Term compared to root gradient | Mean cosine | Fraction opposing root |
| --- | ---: | ---: |
| Current joint_velocity | +0.539 | 0% |
| True GT velocity diagnostic | -0.252 | 72.2% |
| Relative-displacement diagnostic | -0.360 | 70.8% |
| Foot contact | -0.202 | 76.4% |
| Feature reconstruction | +0.005 | 40.3% |
| Latent MSE | +0.001 | 31.9% |

Weighted root latent-gradient norm averages0.0697, latent MSE0.0971, current
velocity0.0054, feature0.0110, foot contact0.0003. Root is not numerically absent.
Direction conflicts exist with some objectives, but foot-gradient scale is small;
cosine alone cannot establish which term controls training. Alternative diagnostics
use unit weight in stored gradient norms, unlike current configured terms.

## 3. Sampler response: preserves injected drift

Coherent shifts rotate/translate positions, poses and vector features consistently;
actor is unchanged except the common-translation negative control.

Root30 checkpoint across12 episodes x3 seeds:

- Clean history: next8-frame root error0.973cm.
- X+30cm: root error30.079cm; first output retains99.99% of the imposed offset
  relative to paired clean output, last output99.64%.
- X-30cm: root error29.941cm; first output retains100.00%, last100.11%.
- Boundary root displacement remains about0.54cm, essentially clean-history size.
- +/-20 degree yaw produces mean absolute heading error20.90/18.64 degrees.
- Baseline shows the same broad lack of offset correction.
- Translating both actors together gives equivariance mean max error8.24e-8m
  for root30. All individual errors passed1e-5m; the perturbation/frame code is
  not introducing a coordinate-system failure.

This rejects a universal immediate aggressive-catch-up response for these tested
states. It does not prove all possible generated states behave this way, nor does
one reference trajectory define the uniquely valid generative future.

## 4. Frozen VAE and oracle root-only optimization

The posterior encoder is given GT future here: this is an oracle diagnostic,
not deployable generation or a fair quality baseline. At clean history its mean
root reconstruction error is0.325cm. With +/-30cm translated history, posterior
mean reconstruction error averages28.006cm despite seeing the correct future.

Optimizing only latent for50 Adam steps (lr0.05), with all decoder parameters
frozen and only root MSE as objective:

| Stage | Root error | First-frame displacement |
| --- | ---: | ---: |
| Posterior mean initialization | 28.006cm | 1.236cm |
| 50 latent optimization steps | 5.342cm | 22.386cm |

Latent change norm averages17.33, a substantial move; this is not an in-distribution
training update. The decoder can express a correction, but its posterior mapping
under drift does not reconstruct that correction well. This is consistent with
a history-conditioned decoder/encoder distribution mismatch, not proof of a hard
architectural impossibility. Both models yield identical decoder-only results,
as expected from their shared frozen VAE and normalization.

## 5. Actual-history oracle interventions

Across12 episodes x3 seeds x2 cutoffs, root30 next8-frame errors:

| History supplied | Root error | Heading error |
| --- | ---: | ---: |
| Original generated history | 29.731cm | 15.53deg |
| Correct only endpoint root by rigid history translation | 6.630cm | 15.29deg |
| Correct only endpoint yaw | 29.089cm | 2.93deg |
| Correct root and yaw | 6.762cm | 2.98deg |
| Full GT history | 0.765cm | 1.25deg |

Root reset improves next-step absolute error largely by removing inherited offset;
this is a diagnostic intervention, not proof of a deployable recovery algorithm.
Yaw correction improves heading but does not solve the dominant position error
over this short horizon. Residual root/pose/history corruption remains after
root+yaw reset. Never interpret reset-branch jumps as a physically executable path.

To isolate local-state effects, replace generated history with GT history rigidly
placed at the SAME generated endpoint position and yaw. Endpoint preservation
errors were <=1.20e-7m and <=2.74e-5degrees. Root30 results:

| Local history | Root error | Boundary displacement | Boundary velocity error | Boundary acceleration magnitude |
| --- | ---: | ---: | ---: | ---: |
| Generated | 29.731cm | 3.494cm | 3.673cm/frame | 3.201cm/frame^2 |
| GT local pose/dynamics at preserved drift | 26.738cm | 0.582cm | 0.284cm/frame | 0.175cm/frame^2 |

Baseline boundary displacement similarly falls4.093 ->0.626cm. This directly
demonstrates that changing local history pose/dynamics can remove most boundary
jumps without removing endpoint root/yaw drift. It does not isolate pose versus
velocity channels, and inherited root offset still dominates global error.

## Causal interpretation and remaining boundary

Confirmed paths:

1. Drifted history + original GT target -> immediate catch-up supervision.
2. Root-only latent fitting -> reduced root error AND larger boundary jump.
3. Drifted/local-corrupted history -> poor next-step position/continuity;
   targeted oracle history interventions change those outcomes.

Not confirmed: root weight30 CAUSED the trained checkpoint's worse global rollout.
The normal sampler does not exhibit the assumed aggressive response under coherent
offset injection. Different training exposure and rollout probability remain
confounders. Root/velocity losses, frozen-VAE reachability, and generated-state
distribution interact; none can be assigned the entire error from these probes.

Prioritize consistent boundary targets and generated-history local dynamics,
then use identical-checkpoint, identical-data controlled training to compare
current versus revised supervision. Include decoder posterior reconstruction
under drift, short displacement/heading, boundary acceleration, long trajectory
metrics, and diverse seeds. Do not simply eliminate all absolute supervision:
that can make persistent drift unpenalized. A deployable drift state cannot use
unknown GT position; it must derive from observable actor-relative geometry and
maintained motion/trajectory state.

## Artifacts, commands and checks

Formal outputs: `outputs/drift_causality_20260927/`, all per-case JSON results,
protocol, summary and diagnosis.png retained. Summary/protocol/checkpoint hashes
copied beside this report; scripts retained as reproduction evidence.

From `/data/autovla/projects`, using the remogen-motion-only interpreter:

```bash
PY=/data/users/autovla/.envs/remogen-motion-only/bin/python
DIR=MoReAct/docs/reports/20260927_drift_causality
CUDA_VISIBLE_DEVICES=5 $PY "$DIR/diagnose.py"
CUDA_VISIBLE_DEVICES=5 $PY "$DIR/decoder_probe.py"
CUDA_VISIBLE_DEVICES=5 $PY "$DIR/history_probe.py"
$PY "$DIR/summarize.py"
$PY "$DIR/target_consistency.py"
```

All five stages exited0. Validated expected row counts, all finite numeric results,
translation equivariance, preserved endpoint root/yaw, analytic target witness,
and repeated original-rollout reproduction across independent probe scripts
(root error difference <1e-6cm). GPU5 returned to4MiB after completion. No temporary
checkpoints or datasets were created; no training assets were deleted or edited.
