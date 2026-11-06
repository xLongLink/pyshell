import ast
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULES = ("model.benchmark", "model.train")


class TestModuleLayout:
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
        ):
            module = import_module(name)
            for command in module.COMMANDS:
                handler = module.COMMANDS[command][0]
                with subtests.test(module=name, command=command):
                    received = []

                    def handle(argv, received=received):
                        received.extend(argv)
                        return 7

                    with patch.object(
                        module,
                        "import_module",
                        return_value=SimpleNamespace(main=handle),
                    ) as load:
                        result = module.main([command, "--help"])
                    assert received == ["--help"]
                    assert result == 7
                    namespace = name.removesuffix(".main") + ".src"
                    load.assert_called_once_with(f"{namespace}.{handler}")

    @pytest.mark.parametrize(
        "name,commands",
        [
            (
                "model.benchmark.main",
                ("list", "run", "report", "evaluate", "audit", "syntax"),
            ),
            (
                "model.train.main",
                ("pipeline", "checkpoint", "generate", "iterate", "experiment"),
            ),
        ],
    )
    def test_retired_commands_are_rejected(self, name, commands, subtests, capsys):
        module = import_module(name)
        for command in commands:
            with subtests.test(command=command), pytest.raises(SystemExit) as error:
                module.main([command])
            assert error.value.code == 2
            assert "invalid choice" in capsys.readouterr().err

    def test_retained_commands_load_real_handlers(self, subtests, capsys):
        for name in (
            "model.benchmark.main",
            "model.train.main",
        ):
            module = import_module(name)
            for command in module.COMMANDS:
                with (
                    subtests.test(module=name, command=command),
                    pytest.raises(SystemExit) as error,
                ):
                    module.main([command, "--help"])
                assert error.value.code == 0
                assert "usage:" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "owner,other", [("benchmark", "train"), ("train", "benchmark")]
    )
    def test_training_and_benchmark_have_no_implementation_dependency(
        self, owner, other
    ):
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
