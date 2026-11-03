"""Check benchmark paths and local environment loading without API calls."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from dataset.benchmark.main import run, parser
from dataset.benchmark.src.backend import ContainerBackend


class EntrypointTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_env_file_is_loaded(self):
        with TemporaryDirectory(dir="/tmp/opencode") as temporary:
            root = Path(temporary)
            (root / ".env").write_text("OPENAI_API_KEY=fixture-key\n", encoding="utf-8")
            args = parser().parse_args(["run", "--model", "test", "--shell-url", "http://testserver"])
            with patch("dataset.benchmark.main.ROOT", root), patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "requires --allow-reset"):
                    await run(args)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "fixture-key")

    async def test_explicit_env_preserves_existing_environment(self):
        with TemporaryDirectory(dir="/tmp/opencode") as temporary:
            env_file = Path(temporary) / "custom.env"
            env_file.write_text("OPENAI_API_KEY=file-key\nOPENAI_BASE_URL=https://example.invalid/v1\n", encoding="utf-8")
            args = parser().parse_args(["run", "--model", "test", "--env-file", str(env_file), "--shell-url", "http://testserver"])
            with patch.dict(os.environ, {"OPENAI_API_KEY": "existing-key"}, clear=True):
                with self.assertRaisesRegex(ValueError, "requires --allow-reset"):
                    await run(args)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "existing-key")
                self.assertEqual(os.environ["OPENAI_BASE_URL"], "https://example.invalid/v1")

    def test_compose_path_stays_at_benchmark_root(self):
        backend = ContainerBackend(8020, Path("unused.log"))
        self.assertEqual(backend.compose, Path(__file__).resolve().parents[1] / "compose.yaml")
        self.assertTrue(backend.compose.is_file())
