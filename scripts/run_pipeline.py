"""Resumable full-data pipeline. Run from any directory with the project environment."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moreact.config import load_config, dump_json
from moreact.train import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--output", default=str(ROOT / "runs/full"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = str(Path(args.config).resolve())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "pipeline.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A pipeline already owns this run directory")
    cfg = load_config(config)
    total = sum(cfg["train"][key] for key in ("stage1_steps", "stage2_steps", "stage3_steps"))
    status = {"pid": os.getpid(), "config": config, "device": args.device,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}

    def run(stage, arguments):
        status.update(stage=stage, status="running", command=arguments,
                      updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        dump_json(output / "status.json", status)
        with (output / (stage + ".log")).open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "moreact"] + arguments,
                                       cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
            status["child_pid"] = process.pid
            dump_json(output / "status.json", status)
            code = process.wait()
        if code:
            status.update(status="failed", exit_code=code)
            dump_json(output / "status.json", status)
            raise SystemExit(code)

    run("prepare", ["prepare", "--config", config, "--device", args.device])
    manifest = json.loads((Path(cfg["data"]["cache"]) / "manifest.json").read_text())
    for kind in ("vae", "diffusion"):
        checkpoint = output / kind / "last.pt"
        command = ["train_" + kind, "--config", config, "--output", str(output / kind), "--device", args.device]
        complete = False
        if checkpoint.exists():
            saved = load_checkpoint(checkpoint)
            if saved["data_digest"] != manifest["digest"]:
                raise ValueError("Completed/resumed checkpoint does not match prepared data")
            complete = saved["step"] >= total
            command += ["--resume", str(checkpoint)]
        elif kind == "diffusion":
            command += ["--vae", str(output / "vae/last.pt")]
        if not complete:
            run("train_" + kind, command)
        run("evaluate_" + kind, ["evaluate", "--checkpoint", str(checkpoint), "--output", str(output / ("eval_" + kind)),
                                  "--split", "val", "--limit", "8", "--frames", "120", "--device", args.device])
    run("preview", ["rollout", "--checkpoint", str(output / "diffusion/last.pt"),
                    "--output", str(output / "preview"), "--split", "val", "--frames", "120", "--device", args.device])
    status.update(status="complete", stage="complete",
                  updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    dump_json(output / "status.json", status)


if __name__ == "__main__":
    main()
