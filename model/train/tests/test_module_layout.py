"""Keep implementation and command routing behind the single module entry points."""

import ast
import sys
import tempfile
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model.train.src.generation.frozen import load_generator

ROOT = Path(__file__).resolve().parents[3]
MODULES = ("model.benchmark", "model.train")


class TestModuleLayout:
    def test_frozen_imports_remain_available_without_obsolete_root_modules(self):
        aliases = (
            "dataset.benchmark.cpu_tasks",
            "model.train.cpu_curriculum",
            "dataset.benchmark.src.serialization",
        )
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "generator.py"
            snapshot.write_text(
                "from dataset.benchmark.cpu_tasks import task\nfrom model.train.cpu_curriculum import feedback\nfrom dataset.benchmark.src.serialization import serialize\nrecord = task('statistics', 0, 713, 'train')\nresult = feedback(\"done('ok')\", {})\nprompt = serialize([{'role': 'user', 'content': 'Sum values.'}])\n"
            )
            namespace = load_generator(snapshot)
        assert namespace["record"]["family"] == "statistics"
        assert namespace["result"]["done"] == "ok"
        assert namespace["prompt"] == "<|task|>\nSum values.\n<|code|>\n"
        assert all(name not in sys.modules for name in aliases)

    def test_only_main_and_package_metadata_live_at_module_roots(self, subtests):
        for name in MODULES:
            with subtests.test(module=name):
                root = ROOT.joinpath(*name.split("."))
                assert (root / "main.py").is_file()
                assert (root / "src").is_dir()
                assert {path.name for path in root.glob("*.py")} <= {
                    "main.py",
                    "__init__.py",
                }
                for path in (root / "src").rglob("*.py"):
                    guards = [
                        node
                        for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.If)
                        and ast.unparse(node.test) == "__name__ == '__main__'"
                    ]
                    assert guards == [], str(path)

    def test_commands_forward_flags_and_exit_codes_to_the_selected_handler(
        self, subtests
    ):
        for name in (
            "model.benchmark.main",
            "model.train.main",
            "model.train.src.generation.cli",
        ):
            module = import_module(name)
            for command, (handler, _) in module.COMMANDS.items():
                with subtests.test(module=name, command=command):
                    received = []

                    def handle(argv):
                        received.extend(argv)
                        return 7

                    with patch.object(
                        module,
                        "import_module",
                        return_value=SimpleNamespace(main=handle),
                    ) as load:
                        result = module.main([command, "--help"])
                    prefix = (
                        [command]
                        if name == "model.benchmark.main" and handler == "cli"
                        else []
                    )
                    assert received == [*prefix, "--help"]
                    assert result == 7
                    namespace = (
                        "model.train.src.generation"
                        if name == "model.train.src.generation.cli"
                        else name.removesuffix(".main") + ".src"
                    )
                    load.assert_called_once_with(f"{namespace}.{handler}")

    def test_benchmark_has_no_training_implementation_dependency(self):
        for path in (ROOT / "model/benchmark/src").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                else:
                    continue
                assert not any(
                    name == "model.train"
                    or name.startswith("model.train.")
                    or name == "train"
                    or name.startswith("train.")
                    for name in names
                ), str(path)
