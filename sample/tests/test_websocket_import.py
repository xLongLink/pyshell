import os
import sys
import subprocess
from pathlib import Path


def test_sample_import_does_not_require_installed_backend():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util; "
            "assert importlib.util.find_spec('api') is None; "
            "from sample.main import run_websocket; "
            "assert run_websocket.__module__ == 'src.api.src.api.websocket'",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "OPENAI_API_KEY": "test-key"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
