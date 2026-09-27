# MoReAct denoising objective

These are MoReAct definitions inspired by the supplied dashboard labels, not a
reproduction of that unknown project's formulas. The CVAE objective is unchanged.
Use `configs/diffusion_geometry.yaml` for the new diffusion objective. Omitting
`diffusion_loss` retains the historical latent-MSE-only objective.

As of 2026-09-27, the current geometry recipe omits `root_position`. Its optional
implementation remains available for historical experiments; archived run configs
and checkpoints retain their original weights. Changing a historical root30
experiment to this objective requires a new run with the changed configuration.

| Config key | Weight | Definition |
| --- | ---: | --- |
| latent_mse | 1 | MSE of predicted vs encoded latent after latent-scale normalization |
| feature_rec | 1 | Huber reconstruction of normalized 276D reactor features, with root channels weighted 5x in the current recipe |
| smpl_joints_rec | 20 | Huber between predicted SMPL-X FK joints and target joints, metres |
| joint_fk_consistency | 10 | Huber between directly predicted joints and their SMPL-X FK joints |
| joint_velocity | 100 | Huber between predicted and target joint displacements, metres/frame (VEL_reactor) |
| bone_length | 10 | Huber between direct-joint bone lengths and target lengths (BL_reactor) |
| foot_contact | 30 | Penalize direct-joint foot displacement under the GT contact mask (FC_reactor) |
| root_orientation | 1 | MSE between predicted and GT root rotation matrices (RO_reactor) |
| root_position | disabled (historically 30) | Optional historical MSE of denormalized SMPL translation, averaged over batch, future frames and XYZ, metres squared |
| root_angular_velocity | 10 | MSE of consecutive relative root rotation matrices (RO_vel_reactor) |
| distance_map | 1 | Huber reconstruction of actor-GT to reactor-FK distances on GT joint pairs closer than 1.5 m |
| joint_contact | 10 | Distance matching on GT joint pairs closer than 0.10 m (JC) |
| root_relative_translation | optional 30 in study B | Huber of predicted FK pelvis displacement versus GT pelvis displacement at future frames 1, 4, 8, each measured from its own final history pelvis, metres |
| root_relative_orientation | optional 10 in study B | MSE of root rotation relative to each side's final history root at future frames 1, 4, 8 |

Huber uses delta=1. Total is the sum of weighted terms; both raw and `weighted_*`
terms are logged. These are initial weights, not tuned optimum values. History
boundary frames are included in velocity/angular losses. Velocities are per-frame
at 30 FPS, not per-second. Latent/simple are one objective, not two added copies.

## Current recipe options and history augmentation

`diffusion_loss_options.feature_root_weight: 5` multiplies elementwise Huber
errors for 24 normalized channels: TRANSL (3), root rotation (6), DTRANS (3),
DROT (6), pelvis JOINTS (3), and pelvis DJOINTS (3). Other channels retain weight
1. The denominator remains B*F*276, not the sum of channel weights. Logged
`feature_rec` already includes this internal weighting; its outer loss weight is 1.
CVAE reconstruction is unchanged. Omitting the option preserves uniform weighting.

`diffusion_loss_options.distance_map_threshold_m: 1.5` selects pairs using
GT distances, not predicted distances, so moving away cannot evade the loss.
The loss is the mean over selected pairs (global selected-element mean under DDP).
Empty masks give a differentiable zero. Omitting this option preserves the
historical full distance map. `joint_contact` retains its separate 0.10 m cutoff.
InterGen's local implementation uses a 1 m predicted-distance mask; the GT mask
here is a deliberate difference.

`history_augmentation` configures diffusion training only: `probability: 0.25`,
`std_m: 0.02`, `max_m: 0.05`. Each selected sample/primitive receives one Gaussian
XY translation, radially clipped to 5 cm, shared across every reactor history
frame. TRANSL and all joint positions shift together; heights, poses and internal
velocity channels stay unchanged. Actor history is untouched. Corruption happens
in world coordinates before constructing the shared history reference frame.
It works with either GT or generated history and does not mutate dataset tensors.

Future positions and poses remain the original GT. The current recipe sets
`diffusion_loss_options.recompute_target_deltas: true`: the first future
DTRANS/DJOINTS/DROT are recomputed against the supplied history before
normalization/encoding for every sample, including generated history without
noise and rollout validation. Boundary position and delta targets therefore
agree regardless of the augmentation mask. Legacy configs omitting this option
retain their old behavior (only noise-selected samples repair the boundary).
The same
corrupted condition goes to encoder, denoiser and decoder. This teaches small
offset recovery toward GT, and can still reward rapid first-frame correction;
it is not smooth-recovery trajectory supervision or evidence of reduced drift.
No corruption occurs during validation/inference or CVAE training. Omitting the
section disables augmentation without consuming RNG. Ordinary resume rejects
changes/removal of either options section; use a new run and `--new-objective`
when intentionally changing the objective from a saved checkpoint.

The inspected DART training uses generated-history curriculum, not explicit
history translation noise. The inspected InterGen training uses motion diffusion
and condition dropout, not this history augmentation. Implementation evidence and
tests: [2026-09-27 report](reports/20260927_history_augmentation/README.md).
The subsequent [diffusion review](reports/20260927_diffusion_review/README.md)
reduced the probability from 50% to 25% and corrected generated-history boundary
targets plus single-device rollout validation and learning-rate restart handling.

When enabled in historical configs, `root_position` explicitly supervises each future root position without dilution
across all 276 feature channels. Prediction and target share the same history
frame and body shape, so their SMPL translation difference equals their pelvis
position difference; XYZ MSE is invariant to the shared rigid frame transform.
It also applies when the frame comes from generated history. This is local
denoising supervision, not backpropagation through the complete sampled rollout.
For a constant 0.30 m Euclidean position error its raw value is 0.03 and its
weighted value is 0.90. Weight 30 is an initial experiment setting, not a measured
optimum. Omitting the key or setting it to zero preserves the previous objective.
Existing run configs/checkpoints are unchanged; running processes do not acquire
this loss automatically. Changed objectives require a separate experiment;
ordinary resume rejects changed diffusion weights, and old best-loss values are
not comparable to the new objective.

The short rollout-drift study branches from the same step-27000 checkpoint.
Branch A uses the current recipe. Branch B adds the two relative-root terms above;
its GT history is transformed into the same reference frame as the generated
history, then each trajectory is differenced against its own history endpoint.
An inherited world translation therefore does not become a one-frame catch-up
target in these two terms. Branch C keeps A's losses and extends feedback from
4 to 8 primitives. These additions affect training only; inference still uses
observed actor history, generated reactor history and text.

`smpl_joints_rec` compares joints in the shared frame defined by the last reactor
history frame (pelvis XY origin, ground-preserving Z, hip-defined heading). It
includes SMPL translation and does not subtract each future frame's pelvis.
Consequently it supervises root position as well as pose. Normalized feature
reconstruction also supervises translation and joint positions. Joint velocity
includes the fixed history boundary; distance/contact losses constrain reactor
placement relative to actor labels. Bone length and root rotation alone cannot
anchor absolute translation.

Only reactor is generated: actor-only BL/FC/VEL/RO terms would have no gradient
to the model. Action_vel is not separately added because joint velocity and root
angular change already provide temporal supervision. RO_inter is not added:
with the same observed actor rotation on both sides, relative-rotation matrix MSE
duplicates reactor root-rotation MSE (orthogonal invariance). Redundant feature
delta consistency terms remain in CVAE training but are not added again here.

Actor future GT is used ONLY as an interaction-loss label, transformed by the
same history-defined coordinate frame and detached. Neither the denoiser nor
CVAE receives actor future as a condition. The frozen decoder retains autograd
with respect to its latent input; its parameters and body-model parameters do
not update. Joint contact is a sparse joint-proximity proxy, not mesh collision
or penetration; samples without GT contact contribute differentiable zero.

Historical validation (before the recipe changes above): 20 automated tests passed, including decoder-input gradients, frozen
VAE weights, actor-label detachment, contact-distance matching, weighted-total
accounting and rejection of resume with changed diffusion weights. Production
batch=128, four primitives passed teacher-history and generated-history
forward/backward checks. Preflight loss was 1.7781 / 2.5949 respectively; these
are initialization checks, not quality measurements.
