"""Exercise scratch cleanup without launching models or GPU training."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("run_smoke", Path(__file__).resolve().parents[1] / "scripts/run_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.mark.parametrize("failure", [False, True])
def test_smoke_archives_before_cleanup_and_retains_failures(tmp_path, failure):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/smoke.yaml").write_text("data:\n  cache: old-cache\n")
    formal = tmp_path / "runs/formal/last.pt"
    formal.parent.mkdir(parents=True)
    formal.write_bytes(b"preserve")
    calls = []

    class Process:
        def __init__(self, command, **kwargs):
            calls.append(command)
            kwargs["stdout"].write("simulated command\n")

        def wait(self):
            return 7 if failure else 0

    original_remove = smoke.shutil.rmtree

    def remove(work):
        report = tmp_path / "docs/reports" / work.name
        record = json.loads((report / "run.json").read_text())
        assert record["status"] == "passed"
        assert record["evidence"]
        assert (report / "evidence/config.yaml").exists()
        original_remove(work)

    with patch.object(smoke.subprocess, "Popen", Process), patch.object(smoke.shutil, "rmtree", remove):
        result = smoke.run(tmp_path, "python")
    report = next((tmp_path / "docs/reports").iterdir())
    record = json.loads((report / "run.json").read_text())
    assert result == (7 if failure else 0)
    assert Path(record["work_dir"]).exists() == failure
    assert record["cleanup"] == ("retained" if failure else "deleted")
    assert len(calls) == (1 if failure else 7)
    assert formal.read_bytes() == b"preserve"


def test_smoke_archive_failure_never_deletes_work(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/smoke.yaml").write_text("data:\n  cache: old-cache\n")
    with patch.object(smoke.subprocess, "Popen") as process, \
            patch.object(smoke, "archive_evidence", side_effect=OSError("disk full")):
        process.return_value.wait.return_value = 0
        assert smoke.run(tmp_path, "python") == 1
    report = next((tmp_path / "docs/reports").iterdir())
    record = json.loads((report / "run.json").read_text())
    assert record["status"] == "archive_or_cleanup_failed"
    assert Path(record["work_dir"]).is_dir()
