# Attribution

MoReAct adapts the autoregressive conditional motion VAE, skip Transformer,
latent diffusion (x0 prediction, cosine schedule), text conditioning and rollout
curriculum designs of DART / DartControl, https://github.com/zkf1997/DART.
The upstream Apache-2.0 license is included as LICENSE. The skip-encoder pattern
also follows the DETR-derived implementation credited in DART to Facebook, Inc.
The local DART reference commit was `bb67ae1ed6ce051080468bf15bc6e54a6c3f8417`.

MoReAct's data pipeline, causal features, two-person conditioning, training,
streaming interface and evaluation are implemented locally. No DART/ReMoGen
motion weights are loaded. CLIP uses pretrained OpenAI weights under its own
license. SMPL-X models and Inter-X data are external assets governed by their
respective licenses and are not distributed in this project.

The optional understanding branch's EMA residual quantizer is adapted from
`ttr_remogen_bridge/src/ttr_remogen/semantic/vq.py` (EMAReset and ResidualQuantizer),
derived from Think-Then-React / OpenMotionLab. The MIT attribution is retained in
`moreact/semantics/TTR_LICENSE`. No bridge motion encoder/decoder or ReMoGen
generator is imported. T5 weights and tokenizer assets retain their own licenses.
