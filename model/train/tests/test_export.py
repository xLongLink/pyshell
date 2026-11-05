import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model.train.src import export


def test_export_resolves_paths_and_preserves_the_converter_exit_code(tmp_path):
    adapter = tmp_path / "unused/../adapter"
    output = tmp_path / "unused/../adapter.gguf"
    root = Path(__file__).resolve().parents[1]
    with patch.object(
        export.subprocess, "run", return_value=SimpleNamespace(returncode=7)
    ) as run:
        assert export.main(["--adapter", str(adapter), "--output", str(output)]) == 7
    run.assert_called_once_with(
        [
            sys.executable,
            str(root / "artifacts/llama.cpp-b12038/convert_lora_to_gguf.py"),
            "--base",
            str(root / "artifacts/base"),
            "--outfile",
            str(tmp_path / "adapter.gguf"),
            str(tmp_path / "adapter"),
        ],
        check=False,
    )
