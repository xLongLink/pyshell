"""Preserve frozen CPU evaluation's GGUF server settings."""

import pytest
from types import SimpleNamespace
from unittest.mock import patch
from model.benchmark.src import cpu_evaluate
from model.benchmark.src.inference_server import gguf_command


@pytest.mark.parametrize(("port", "context_size", "batch_threads"), [(8030, 8192, 12), (8032, 4096, 8)])
def test_gguf_command_preserves_cpu_runtime_options(tmp_path, port, context_size, batch_threads):
    model = tmp_path / "base.gguf"
    assert gguf_command(tmp_path, model, port=port, context_size=context_size, batch_threads=batch_threads) == [
        str(tmp_path / "llama-b12038/llama-b12038/llama-server"),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context_size),
        "--parallel",
        "1",
        "--threads",
        "8",
        "--threads-batch",
        str(batch_threads),
        "--n-gpu-layers",
        "0",
    ]


@pytest.mark.parametrize("with_adapter", [False, True])
def test_cpu_evaluation_keeps_native_completion_options(tmp_path, with_adapter):
    adapter = tmp_path / "adapter.gguf" if with_adapter else None
    args = SimpleNamespace(backend="gguf", gguf_base=tmp_path / "base.gguf", adapter=adapter, output=tmp_path)
    with patch.object(cpu_evaluate, "inference_server") as server:
        with cpu_evaluate.generator(args) as generate:
            assert generate is cpu_evaluate.completion
    command = server.call_args.args[0]
    assert command[command.index("--model") + 1] == str(args.gguf_base)
    assert command[command.index("--port") + 1] == "8032"
    assert command[command.index("--ctx-size") + 1] == "4096"
    assert command[command.index("--threads-batch") + 1] == "8"
    assert "--jinja" not in command
    assert "--reasoning" not in command
    assert "--api-key" not in command
    if with_adapter:
        assert command[command.index("--lora") + 1] == str(adapter)
    else:
        assert "--lora" not in command
