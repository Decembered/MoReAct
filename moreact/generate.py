from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import dump_json
from .data import condition_window
from .diffusion import LatentDiffusion
from .geometry import BodyModels, GENDERS, POSE, TRANSL, rotation_6d_to_matrix, transform_features
from .models import ReactionDenoiser, ReactionVAE
from .train import load_checkpoint


@dataclass
class ReactionState:
    reactor_history: torch.Tensor
    betas: torch.Tensor
    genders: torch.Tensor
    offsets: torch.Tensor


class ReactionGenerator:
    """Streaming API. step() only accepts observed actor history, never future."""

    def __init__(self, checkpoint, device="cpu", body_models=None, seed=0, guidance=1.):
        saved = load_checkpoint(checkpoint)
        if saved["kind"] != "diffusion":
            raise ValueError("Generation requires a diffusion checkpoint")
        self.cfg = saved["config"]
        self.device = torch.device(device)
        torch.set_num_threads(self.cfg["train"]["threads"])
        self.model = ReactionDenoiser(self.cfg).to(self.device).eval().requires_grad_(False)
        self.model.load_state_dict(saved["ema"])
        self.vae = ReactionVAE(self.cfg).to(self.device).eval().requires_grad_(False)
        self.vae.load_state_dict(saved["vae"])
        self.stats = {k: v.to(self.device) for k, v in saved["stats"].items()}
        self.diffusion = LatentDiffusion(self.cfg["model"]["diffusion_steps"]).to(self.device)
        self.body = BodyModels(body_models or self.cfg["data"]["body_models"], self.device)
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.guidance = guidance
        self.data_digest = saved["data_digest"]
        self.checkpoint_step = saved["step"]

    def initialize(self, reactor_history, betas, genders, offsets):
        H = self.cfg["data"]["history"]
        if reactor_history.ndim != 3 or reactor_history.shape[1:] != (H, 276):
            raise ValueError("Expected [B, history, 276] initial reactor motion")
        return ReactionState(*(x.to(self.device).clone() for x in (reactor_history, betas, genders, offsets)))

    @torch.no_grad()
    def step(self, actor_history, state, text_embedding=None, actor_mode="normal"):
        actor_history = actor_history.to(self.device)
        if actor_history.shape != state.reactor_history.shape:
            raise ValueError("Actor history must match [B, history, 276]")
        if not torch.isfinite(actor_history).all() or not torch.isfinite(state.reactor_history).all():
            raise ValueError("History contains nonfinite values")
        if actor_mode not in ("normal", "shuffle", "remove"):
            raise ValueError("Unknown actor condition diagnostic")
        a, r, origin, basis = condition_window(actor_history, state.reactor_history, state.offsets, self.stats)
        if actor_mode == "remove":
            a = torch.zeros_like(a)
        elif actor_mode == "shuffle":
            # Fixed temporal permutation of already observed frames; no future access.
            a = a.flip(1)
        text = (torch.zeros(len(a), 512, device=self.device) if text_embedding is None
                else text_embedding.to(self.device))
        if text.shape != (len(a), 512):
            raise ValueError("Expected [B, 512] CLIP embedding")
        z = self.diffusion.sample(self.model, a, r, text, self.cfg["model"]["latent_dim"],
                                  self.guidance, self.rng)
        prediction = self.vae.decode(z * self.vae.latent_scale, a, r)
        local = prediction * self.stats["std"] + self.stats["mean"]
        world = transform_features(local, origin, basis, state.offsets[:, 1], inverse=True)
        world = self.body.repair(world, state.reactor_history, state.betas[:, 1], state.genders[:, 1])
        if not torch.isfinite(world).all():
            raise FloatingPointError("Nonfinite generated motion")
        H = self.cfg["data"]["history"]
        updated = torch.cat((state.reactor_history, world), 1)[:, -H:]
        return world, ReactionState(updated, state.betas, state.genders, state.offsets)


@torch.no_grad()
def rollout_arrays(generator, actor_observations, initial_reactor, betas, genders, offsets,
                   frames, text_embedding=None, actor_mode="normal"):
    """Only initial_reactor is accepted: reactor target futures cannot enter inference."""
    H, F = generator.cfg["data"]["history"], generator.cfg["data"]["future"]
    if frames <= 0 or len(actor_observations) < H + frames:
        raise ValueError("Need actor observations covering history plus requested duration")
    state = generator.initialize(initial_reactor[None], betas[None], genders[None], offsets[None])
    # Warm body model loading is excluded from network/FK latency, explicitly reported.
    generator.body.model(GENDERS[int(genders[1])])
    result, latencies = [], []
    produced = 0
    while produced < frames:
        cutoff = H + produced
        actor = actor_observations[cutoff - H:cutoff][None]
        if generator.device.type == "cuda":
            torch.cuda.synchronize(generator.device)
        start = time.perf_counter()
        block, state = generator.step(actor, state, text_embedding, actor_mode)
        if generator.device.type == "cuda":
            torch.cuda.synchronize(generator.device)
        latencies.append(time.perf_counter() - start)
        take = min(F, frames - produced)
        result.append(block[0, :take].cpu())
        produced += take
    return torch.cat(result), latencies


def load_episode(cache, episode=None, split="test"):
    cache = Path(cache)
    manifest = json.loads((cache / "manifest.json").read_text())
    matches = [r for r in manifest["records"] if (r["episode"] == episode if episode else r["split"] == split)]
    if not matches:
        raise ValueError("Episode not found in prepared cache")
    record = matches[0]
    with np.load(cache / record["file"]) as z:
        data = {k: z[k] for k in z.files}
    return record, data, manifest


def export_smpl(path, features, betas, gender, fps):
    from scipy.spatial.transform import Rotation
    rotations = rotation_6d_to_matrix(torch.as_tensor(features)[..., POSE].reshape(-1, 22, 6)).numpy()
    body = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 66)
    poses = np.zeros((len(body), 165), dtype=np.float32)
    poses[:, :66] = body
    np.savez_compressed(path, poses=poses, trans=features[:, TRANSL], betas=betas,
                        gender=GENDERS[int(gender)], mocap_framerate=fps, coordinate_system="Z-up")


def rollout(checkpoint, output, cache=None, episode=None, split="test", frames=120,
            device="cpu", seed=0, guidance=1., no_text=False, text=None,
            actor_mode="normal", video=True, body_models=None, start_frame=0):
    from .evaluate import motion_metrics
    generator = ReactionGenerator(checkpoint, device, body_models, seed, guidance)
    cache = cache or generator.cfg["data"]["cache"]
    record, data, manifest = load_episode(cache, episode, split)
    if manifest["digest"] != generator.data_digest:
        raise ValueError("Checkpoint/data mismatch")
    H = generator.cfg["data"]["history"]
    if start_frame < 0:
        raise ValueError("start_frame must be nonnegative")
    remaining = data["features"].shape[1] - start_frame - H
    frames = min(frames, remaining)
    if frames <= 0:
        raise ValueError("No future frames available")
    actor = torch.from_numpy(data["features"][0, start_frame:start_frame + H + frames].copy())
    reactor_initial = torch.from_numpy(data["features"][1, start_frame:start_frame + H].copy())
    caption = "" if no_text else (text if text is not None else str(data["captions"][0]))
    embedding = None
    if not no_text:
        if text is not None:
            from .text import FrozenCLIP
            embedding = FrozenCLIP(generator.cfg["data"]["clip_model"], device).encode([text])
        else:
            embedding = torch.from_numpy(data["text_embeddings"][:1].copy())
    prediction, latency = rollout_arrays(generator, actor, reactor_initial,
                                          torch.from_numpy(data["betas"]), torch.from_numpy(data["genders"]),
                                          torch.from_numpy(data["offsets"]), frames, embedding, actor_mode)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pred = prediction.numpy()
    actor_future = actor[H:].numpy()
    # Future targets are read only after generation, for evaluation/export.
    gt = data["features"][1, start_frame + H:start_frame + H + frames]
    np.savez_compressed(output / "motion.npz", actor=actor_future, reactor=pred, target=gt,
                        initial_reactor=reactor_initial.numpy(), betas=data["betas"],
                        genders=data["genders"], fps=generator.cfg["data"]["fps"])
    for name, values, idx in (("actor", actor_future, 0), ("reactor", pred, 1), ("target", gt, 1)):
        export_smpl(output / (name + "_smplx.npz"), values, data["betas"][idx], data["genders"][idx],
                    generator.cfg["data"]["fps"])
    metadata = {"episode": record, "checkpoint": str(checkpoint), "checkpoint_step": generator.checkpoint_step,
                "seed": seed, "frames": frames, "start_frame": start_frame,
                "generation_start": start_frame + H, "text": caption, "actor_mode": actor_mode,
                "guidance": guidance, "config": generator.cfg,
                "segment_seconds": latency, "mean_segment_seconds": float(np.mean(latency)),
                "p95_segment_seconds": float(np.percentile(latency, 95)),
                "latency_excludes": "checkpoint, body-model loading, text encoding, rendering and file I/O",
                "metrics": motion_metrics(pred, gt, actor_future, generator.cfg["data"]["fps"],
                                          generator.cfg["data"]["future"], reactor_initial[-1].numpy())}
    dump_json(output / "metadata.json", metadata)
    if video:
        from .render import render_video
        render_video(output / "motion.npz", output / "preview.mp4", caption)
    print(json.dumps({"output": str(output), "frames": frames,
                      "mean_segment_seconds": metadata["mean_segment_seconds"], "metrics": metadata["metrics"]}), flush=True)
    return metadata
