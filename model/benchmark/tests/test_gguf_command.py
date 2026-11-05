"""Preserve frozen CPU evaluation's GGUF server settings."""

import pytest
from types import SimpleNamespace
from unittest.mock import patch
from model.benchmark.src import cpu_evaluate


@pytest.mark.parametrize("with_adapter", [False, True])
def test_cpu_evaluation_keeps_native_completion_options(tmp_path, with_adapter):
    adapter = tmp_path / "adapter.gguf" if with_adapter else None
    args = SimpleNamespace(backend="gguf", gguf_base=tmp_path / "base.gguf", adapter=adapter, output=tmp_path)
    with patch.object(cpu_evaluate, "inference_server") as server:
        with cpu_evaluate.generator(args) as generate:
            assert generate is cpu_evaluate.completion
    command = server.call_args.args[0]
    assert command == [
        str(cpu_evaluate.ROOT / "local/llama-b12038/llama-b12038/llama-server"),
        "--model",
        str(args.gguf_base),
        "--host",
        "127.0.0.1",
        "--port",
        "8032",
        "--ctx-size",
        "4096",
        "--parallel",
        "1",
        "--threads",
        "8",
        "--threads-batch",
        "8",
        "--n-gpu-layers",
        "0",
        *(["--lora", str(adapter)] if with_adapter else []),
    ]
    assert server.call_args.kwargs == {"port": 8032}
