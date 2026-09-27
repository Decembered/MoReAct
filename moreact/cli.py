from __future__ import annotations

import argparse
from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="MoReAct: causal Inter-X reaction motion generation")
    sub = parser.add_subparsers(dest="command", required=True)
    from .semantics.cli import add_parser
    add_parser(sub)
    p = sub.add_parser("prepare", help="Prepare raw Inter-X, CLIP embeddings and train-only statistics")
    p.add_argument("--config")
    p.add_argument("--limit", type=int, help="Maximum episodes per split (use a separate cache)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reuse-cache", help="Reuse compatible episode features after checking source digests")
    for name in ("train_vae", "train_diffusion"):
        p = sub.add_parser(name)
        p.add_argument("--config")
        p.add_argument("--output", required=True)
        p.add_argument("--steps", type=int, help="Absolute target step; default is the whole curriculum")
        p.add_argument("--resume")
        p.add_argument("--device")
        p.add_argument("--overfit", action="store_true", help="Repeat the first batch of train windows")
        if name == "train_diffusion":
            p.add_argument("--vae", help="VAE checkpoint (not needed when resuming)")
    p = sub.add_parser("rollout")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cache")
    p.add_argument("--episode")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--guidance", type=float, default=1.)
    text = p.add_mutually_exclusive_group()
    text.add_argument("--no-text", action="store_true")
    text.add_argument("--text")
    p.add_argument("--actor-mode", choices=["normal", "shuffle", "remove"], default="normal")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--body-models")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cache")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--limit", type=int, default=8, help="0 evaluates the entire split")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ablations", action="store_true")
    p.add_argument("--video", action="store_true")
    args = parser.parse_args()
    kwargs = vars(args).copy()
    command = kwargs.pop("command")
    if command == "semantics":
        from .semantics.cli import main as semantic_main
        semantic_main(kwargs)
    elif command == "prepare":
        import torch
        from .data import prepare
        cfg = load_config(kwargs.pop("config"))
        torch.set_num_threads(cfg["train"]["threads"])
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        prepare(cfg, **kwargs)
    elif command.startswith("train_"):
        from .train import Trainer
        cfg = load_config(kwargs.pop("config"))
        device = kwargs.pop("device")
        if device:
            cfg["train"]["device"] = device
        steps = kwargs.pop("steps")
        if "vae" in kwargs:
            kwargs["vae_path"] = kwargs.pop("vae")
        Trainer(cfg, command[len("train_"):], **kwargs).run(steps)
    elif command == "rollout":
        from .generate import rollout
        kwargs["video"] = not kwargs.pop("no_video")
        rollout(**kwargs)
    else:
        from .evaluate import evaluate
        kwargs["ablations"] = not kwargs.pop("no_ablations")
        evaluate(**kwargs)
