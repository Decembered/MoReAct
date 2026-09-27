"""Offline semantic commands; no language dependency is imported by --help."""
from pathlib import Path


def add_parser(sub):
    parser = sub.add_parser("semantics", help="Frozen shared-CVAE / RVQ whole-motion understanding")
    commands = parser.add_subparsers(dest="semantic_command", required=True)
    p = commands.add_parser("cache-latents")
    p.add_argument("--checkpoint", required=True, help="Diffusion checkpoint with embedded CVAE")
    p.add_argument("--cache", required=True, help="Prepared MoReAct data cache")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=0, help="Per split; 0 means complete cache")
    p.add_argument("--batch-size", type=int, default=64)
    p = commands.add_parser("fit-rvq")
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024, help="Paired windows per EMA update")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--levels", type=int, choices=[1, 2], default=2)
    p.add_argument("--decay", type=float, default=.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume")
    p = commands.add_parser("cache-tokens")
    p.add_argument("--cache", required=True)
    p.add_argument("--rvq", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p = commands.add_parser("train")
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", required=True, type=int, help="Explicit absolute update budget")
    p.add_argument("--base-model", default="/data/autovla/projects/models/flan-t5-large")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume")
    p = commands.add_parser("describe")
    p.add_argument("--motion-checkpoint", required=True)
    p.add_argument("--rvq", required=True)
    p.add_argument("--checkpoint", required=True, help="Captioner checkpoint")
    p.add_argument("--input", required=True, help="MoReAct paired NPZ (features,betas,genders,offsets)")
    p.add_argument("--output", required=True, help="New result JSON path")
    p.add_argument("--device", default="cpu")
    p.add_argument("--motion-device", default="cpu")
    p = commands.add_parser("evaluate")
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rvq", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--motion-checkpoint", help="Also compute both-role continuous/quantized FK reconstruction")
    p.add_argument("--motion-cache", help="Prepared source poses, required with --motion-checkpoint")
    return parser


def main(kwargs):
    import json
    import torch
    from ..config import dump_json
    # Small motion windows benefit from bounded CPU threads; do not alter CUDA RNG.
    torch.set_num_threads(4)
    command = kwargs.pop("semantic_command")
    if command == "cache-latents":
        from .cache import cache_latents
        result = cache_latents(**kwargs)
        result = dict(output=kwargs["output"], counts=result["counts"], identity=result["identity"])
    elif command == "fit-rvq":
        from .cache import fit_rvq
        result = fit_rvq(**kwargs)
    elif command == "cache-tokens":
        from .cache import cache_tokens
        result = cache_tokens(**kwargs)
        result = dict(output=kwargs["output"], counts=result["counts"], rvq_id=result["rvq_id"])
    elif command == "train":
        from .language import train_captioner
        result = train_captioner(**kwargs)
    elif command == "describe":
        import numpy as np
        from .tokenizer import SharedMotionTokenizer
        from .language import InteractionCaptioner
        output = Path(kwargs["output"])
        if output.exists():
            raise FileExistsError(output)
        motion = SharedMotionTokenizer.from_checkpoint(kwargs["motion_checkpoint"], kwargs["motion_device"])
        captioner = InteractionCaptioner(motion, kwargs["rvq"], kwargs["checkpoint"], kwargs["device"])
        with np.load(kwargs["input"], allow_pickle=False) as data:
            result = captioner.describe(*(data[k] for k in ("features", "betas", "genders", "offsets")))
        dump_json(output, result)
    else:
        from .language import evaluate_captioner
        motion_checkpoint, motion_cache = kwargs.pop("motion_checkpoint"), kwargs.pop("motion_cache")
        if bool(motion_checkpoint) != bool(motion_cache):
            raise ValueError("Specify both --motion-checkpoint and --motion-cache for reconstruction")
        result = evaluate_captioner(**kwargs)
        if motion_checkpoint:
            from .common import read_manifest
            from .diagnostics import reconstruction_report
            report = reconstruction_report(motion_checkpoint, kwargs["rvq"], motion_cache,
                                           read_manifest(kwargs["cache"], "tokens"), kwargs["split"], kwargs["limit"], kwargs["device"])
            dump_json(Path(kwargs["output"]) / "reconstruction.json", report)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
