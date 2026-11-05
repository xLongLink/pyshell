"""Check benchmark paths and local environment loading without API calls."""

import os
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch
from model.benchmark.src.cli import run, parser
from model.benchmark.src.runner import Limits
from model.benchmark.src.backend import ContainerBackend


class TestEntrypoint:
    def test_default_limits_match_runner(self):
        args = parser().parse_args(["run", "--model", "test"])
        assert Limits(seconds=args.max_seconds, calls=args.max_calls, tokens=args.max_trial_tokens) == Limits()

    def test_explicit_limits_override_defaults(self):
        args = parser().parse_args(["run", "--model", "test", "--max-seconds", "45", "--max-calls", "5", "--max-trial-tokens", "9000"])
        assert Limits(seconds=args.max_seconds, calls=args.max_calls, tokens=args.max_trial_tokens) == Limits(45, 5, 9000)

    @pytest.mark.anyio
    async def test_default_env_file_is_loaded(self):
        with TemporaryDirectory(dir="/tmp/opencode") as temporary:
            root = Path(temporary)
            (root / ".env").write_text("OPENAI_API_KEY=fixture-key\n", encoding="utf-8")
            args = parser().parse_args(["run", "--model", "test", "--shell-url", "http://testserver"])
            with patch("model.benchmark.src.cli.ROOT", root), patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="requires --allow-reset"):
                    await run(args)
                assert os.environ["OPENAI_API_KEY"] == "fixture-key"

    @pytest.mark.anyio
    async def test_explicit_env_preserves_existing_environment(self):
        with TemporaryDirectory(dir="/tmp/opencode") as temporary:
            env_file = Path(temporary) / "custom.env"
            env_file.write_text("OPENAI_API_KEY=file-key\nOPENAI_BASE_URL=https://example.invalid/v1\n", encoding="utf-8")
            args = parser().parse_args(["run", "--model", "test", "--env-file", str(env_file), "--shell-url", "http://testserver"])
            with patch.dict(os.environ, {"OPENAI_API_KEY": "existing-key"}, clear=True):
                with pytest.raises(ValueError, match="requires --allow-reset"):
                    await run(args)
                assert os.environ["OPENAI_API_KEY"] == "existing-key"
                assert os.environ["OPENAI_BASE_URL"] == "https://example.invalid/v1"

    def test_compose_path_stays_at_benchmark_root(self):
        backend = ContainerBackend(8020, Path("unused.log"))
        assert backend.compose == Path(__file__).resolve().parents[1] / "compose.yaml"
        assert backend.compose.is_file()

    @pytest.mark.anyio
    @pytest.mark.parametrize("no_build", [False, True])
    async def test_build_precedes_model_initialization_unless_disabled(self, tmp_path, no_build):
        arguments = ["run", "--model", "test", "--api", "raw", "--output", str(tmp_path)]
        if no_build:
            arguments.append("--no-build")
        args = parser().parse_args(arguments)
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-key"}),
            patch.object(ContainerBackend, "command") as command,
            patch("model.benchmark.src.cli.RawCompletionModel") as model,
        ):

            def initialize(*args, **kwargs):
                assert command.await_args_list == ([] if no_build else [call("build", "pyshell")])
                raise RuntimeError("model initialization failed")

            model.side_effect = initialize
            with pytest.raises(RuntimeError, match="model initialization failed"):
                await run(args)
            expected = [] if no_build else [call("build", "pyshell")]
            assert command.await_args_list == [*expected, call("down", "--remove-orphans")]
