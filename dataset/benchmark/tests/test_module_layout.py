"""Keep implementation and command routing behind the single module entry points."""

import ast
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from importlib import import_module
from unittest.mock import patch
from dataset.generation.src.frozen import load_generator

ROOT = Path(__file__).resolve().parents[3]
MODULES = ("dataset.benchmark", "dataset.generation", "model.train")


class ModuleLayoutTests(unittest.TestCase):
    def test_frozen_imports_remain_available_without_obsolete_root_modules(self):
        aliases = ("dataset.benchmark.cpu_tasks", "model.train.cpu_curriculum")
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "generator.py"
            snapshot.write_text(
                "from dataset.benchmark.cpu_tasks import task\nfrom model.train.cpu_curriculum import feedback\nrecord = task('statistics', 0, 713, 'train')\nresult = feedback(\"done('ok')\", {})\n"
            )
            namespace = load_generator(snapshot)
        self.assertEqual(namespace["record"]["family"], "statistics")
        self.assertEqual(namespace["result"]["done"], "ok")
        self.assertTrue(all(name not in sys.modules for name in aliases))

    def test_only_main_and_package_metadata_live_at_module_roots(self):
        for name in MODULES:
            with self.subTest(module=name):
                root = ROOT.joinpath(*name.split("."))
                self.assertTrue((root / "main.py").is_file())
                self.assertTrue((root / "src").is_dir())
                self.assertLessEqual({path.name for path in root.glob("*.py")}, {"main.py", "__init__.py"})
                for path in (root / "src").glob("*.py"):
                    guards = [
                        node
                        for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
                    ]
                    self.assertEqual(guards, [], str(path))

    def test_commands_forward_flags_and_exit_codes_to_the_selected_handler(self):
        for name in MODULES:
            module = import_module(name + ".main")
            for command, (handler, _) in module.COMMANDS.items():
                with self.subTest(module=name, command=command):
                    received = []

                    def handle(argv):
                        received.extend(argv)
                        return 7

                    with patch.object(module, "import_module", return_value=SimpleNamespace(main=handle)) as load:
                        result = module.main([command, "--help"])
                    prefix = [command] if name == "dataset.benchmark" and handler == "cli" else []
                    self.assertEqual(received, [*prefix, "--help"])
                    self.assertEqual(result, 7)
                    load.assert_called_once_with(f"{name}.src.{handler}")
