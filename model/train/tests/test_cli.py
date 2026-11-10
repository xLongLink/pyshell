import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_benchmark_help_describes_settings_configuration():
    result = subprocess.run(
        [sys.executable, "-m", "model.benchmark.main", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "settings.py" in result.stdout


def test_training_rejects_cli_flags_before_starting_a_run():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "model.train.main", "--max_steps", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "edit model/train/src/settings.py" in result.stderr


@pytest.mark.parametrize(
    "owner,other", [("benchmark", "train"), ("train", "benchmark")]
)
def test_training_and_benchmark_have_no_implementation_dependency(owner, other):
    for path in (ROOT / f"model/{owner}/src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            assert not any(
                name == f"model.{other}"
                or name == other
                or name.startswith((f"model.{other}.", f"{other}."))
                for name in names
            ), str(path)
