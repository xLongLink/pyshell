"""Check adapter export routing without running the converter."""

from types import SimpleNamespace
from unittest.mock import patch

from model.train.src import export


def test_export_resolves_paths_and_preserves_the_converter_exit_code(tmp_path):
    adapter = tmp_path / "adapter"
    output = tmp_path / "adapter.gguf"
    with patch.object(
        export.subprocess, "run", return_value=SimpleNamespace(returncode=7)
    ) as run:
        assert export.main(["--adapter", str(adapter), "--output", str(output)]) == 7
    run.assert_called_once_with(
        export.export_command(adapter.resolve(), output.resolve()), check=False
    )
