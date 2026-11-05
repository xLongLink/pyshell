import sys
import pytest
import subprocess
from pathlib import Path
from contextlib import nullcontext
from model.benchmark.src import cli, backend


def test_cli_preserves_backend_lock_import():
    assert cli.backend_lock is backend.backend_lock


@pytest.mark.parametrize("fail", [False, True])
def test_backend_lock_rejects_other_process_and_releases(tmp_path, monkeypatch, fail):
    monkeypatch.setattr(backend, "ROOT", tmp_path)
    script = (
        "from pathlib import Path\n"
        "from model.benchmark.src import backend\n"
        f"backend.ROOT = Path({str(tmp_path)!r})\n"
        "with backend.backend_lock():\n"
        "    pass\n"
    )

    def contend():
        return subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, check=False
        )

    expected_error = pytest.raises(ValueError, match="body failed") if fail else nullcontext()
    with expected_error, backend.backend_lock():
        assert (tmp_path / "results/backend.lock").is_file()
        result = contend()
        assert result.returncode != 0
        assert "RuntimeError: Another benchmark owns the execution backend" in result.stderr
        assert "BlockingIOError" in result.stderr
        if fail:
            raise ValueError("body failed")
    result = contend()
    assert result.returncode == 0, result.stderr


def test_audit_import_does_not_load_cli():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from model.benchmark.src import audit, backend; "
            "assert 'model.benchmark.src.cli' not in sys.modules; "
            "assert audit.backend_lock is backend.backend_lock",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
