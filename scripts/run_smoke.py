"""Run an isolated smoke, archive small evidence, then remove successful scratch work."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_record(report, record):
    pending = report / "run.json.tmp"
    pending.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    pending.replace(report / "run.json")
    lines = ["# MoReAct smoke", "", "- Status: " + record["status"],
             "- Started: " + record["started_utc"],
             "- Work directory: `" + record["work_dir"] + "`",
             "- Cleanup: " + record["cleanup"], "",
             "Pipeline check only; this does not establish motion quality or real-time performance.",
             "Reproduce from the project root with `bash scripts/smoke.sh` and the recorded Python environment.",
             "Historical scratch paths are provenance, not reusable checkpoint locations.", "",
             "Commands and exit codes: [run.json](run.json). Configs, logs and metrics are archived under `evidence/`.", ""]
    (report / "README.md").write_text("\n".join(lines))


def archive_evidence(work, report):
    """Archive only small textual evidence from our own experiment directories."""
    files = [work / "config.yaml"]
    for directory in ("logs", "data", "runs", "outputs"):
        files.extend(sorted((work / directory).rglob("*")))
    manifest = []
    for source in files:
        if not source.is_file() or source.is_symlink():
            continue
        if source.suffix not in (".json", ".jsonl", ".yaml", ".log"):
            continue
        relative = source.relative_to(work)
        destination = report / "evidence" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(destination.read_bytes()).hexdigest() != source_hash:
            raise OSError("Evidence verification failed: " + str(source))
        manifest.append({"path": str(relative), "bytes": source.stat().st_size, "sha256": source_hash})
    return manifest


def run(root=ROOT, python=sys.executable):
    root = Path(root).resolve()
    scratch = root / "tmp"
    scratch.mkdir(exist_ok=True)
    if scratch.is_symlink():
        raise ValueError("Smoke scratch root must not be a symlink")
    prefix = "smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_"
    work = Path(tempfile.mkdtemp(prefix=prefix, dir=str(scratch)))
    report = root / "docs" / "reports" / work.name
    report.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_utc": utc_now(), "work_dir": str(work),
              "python": python, "cleanup": "pending", "commands": []}
    write_record(report, record)
    print("Smoke work: " + str(work), flush=True)
    print("Smoke report: " + str(report), flush=True)
    code = 1
    try:
        cfg = yaml.safe_load((root / "configs/smoke.yaml").read_text())
        cfg["data"]["cache"] = str(work / "data")
        config = work / "config.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        logs = work / "logs"
        logs.mkdir()
        vae, diffusion = work / "runs/vae", work / "runs/diffusion"
        commands = [
            ("tests", ["-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(work / "pytest")]),
            ("prepare", ["-m", "moreact", "prepare", "--config", str(config), "--limit", "8"]),
            ("train_vae", ["-m", "moreact", "train_vae", "--config", str(config), "--output", str(vae), "--steps", "200"]),
            ("train_diffusion", ["-m", "moreact", "train_diffusion", "--config", str(config),
                                  "--vae", str(vae / "last.pt"), "--output", str(diffusion), "--steps", "200"]),
            ("evaluate_vae", ["-m", "moreact", "evaluate", "--checkpoint", str(vae / "last.pt"),
                              "--output", str(work / "outputs/vae_eval"), "--limit", "2", "--frames", "32"]),
            ("evaluate_diffusion", ["-m", "moreact", "evaluate", "--checkpoint", str(diffusion / "last.pt"),
                                    "--output", str(work / "outputs/diffusion_eval"), "--limit", "2", "--frames", "32"]),
            ("preview", ["-m", "moreact", "rollout", "--checkpoint", str(diffusion / "last.pt"),
                         "--output", str(work / "outputs/preview"), "--frames", "64"]),
        ]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for name, args in commands:
            command = [python] + args
            entry = {"stage": name, "argv": command, "started_utc": utc_now(), "returncode": None}
            record["commands"].append(entry)
            write_record(report, record)
            print(name + ": " + shlex.join(command), flush=True)
            with (logs / (name + ".log")).open("w") as log:
                process = subprocess.Popen(command, cwd=str(root), env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    result = process.wait()
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
            entry.update(returncode=result, finished_utc=utc_now())
            if result:
                code = result if result > 0 else 128 - result
                raise RuntimeError("Stage failed: " + name + "; see " + str(logs / (name + ".log")))
        record["status"] = "passed"
        code = 0
    except KeyboardInterrupt:
        record.update(status="interrupted", error="KeyboardInterrupt")
        code = 130
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        print(str(exc), file=sys.stderr)
    finally:
        record["finished_utc"] = utc_now()
        record["cleanup"] = "retained"
        try:
            record["evidence"] = archive_evidence(work, report)
            write_record(report, record)
            # Only the directory allocated by this invocation is eligible for removal.
            if record["status"] == "passed":
                if work.is_symlink() or scratch.is_symlink() or work.resolve().parent != scratch.resolve():
                    raise ValueError("Unsafe smoke cleanup path")
                shutil.rmtree(work)
                record["cleanup"] = "deleted"
                write_record(report, record)
        except Exception as exc:
            record.update(status="archive_or_cleanup_failed", error=str(exc))
            code = 1
            write_record(report, record)
            print("Archive/cleanup failed: " + str(exc), file=sys.stderr)
    print("Smoke " + record["status"] + "; report: " + str(report), flush=True)
    return code


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(run())
