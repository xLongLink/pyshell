"""Preserve shared corpus defaults and syntax's explicit file override."""

import io
import json
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
from model.benchmark.src import syntax
from pyshell.serialization import CODE, TASK, CONSOLE, END_CODE
from model.benchmark.src.corpus import default_training_data


class TestDefaultTrainingData:
    def test_defaults_are_repository_relative_in_both_modes(self, subtests):
        repo = Path(__file__).resolve().parents[2]
        for serialization, directory in (("raw", "train/data/raw"), ("chat", "train/data/chat")):
            with subtests.test(serialization=serialization):
                assert default_training_data(serialization) == repo / directory

    def test_syntax_reads_default_validation_or_explicit_file_without_resolving(self, subtests):
        repo = Path(__file__).resolve().parents[2]
        record = {
            "id": "example",
            "family": "example",
            "prompt": "Write Python",
            "messages": [{"role": "user", "content": "Write Python"}, {"role": "assistant", "content": "x = 1"}],
        }
        completion = {
            "content": "x = 1",
            "stopping_word": END_CODE,
            "choices": [{"message": {"content": "x = 1"}, "finish_reason": "stop"}],
        }
        for serialization, directory in (("raw", "train/data/raw"), ("chat", "train/data/chat")):
            for override in (None, "custom/../validation.jsonl"):
                with subtests.test(serialization=serialization, override=override), tempfile.TemporaryDirectory() as temporary:
                    output = Path(temporary) / "syntax"
                    argv = ["--model", "fixture", "--serialization", serialization, "--output", str(output)]
                    expected = Path(override) if override else repo / directory / "validation.jsonl"
                    if override is not None:
                        argv.extend(["--data", override])
                    with (
                        patch.object(Path, "read_text", autospec=True, return_value=json.dumps(record)) as read,
                        patch.object(syntax.urllib.request, "urlopen", return_value=io.StringIO(json.dumps(completion))) as urlopen,
                        redirect_stdout(io.StringIO()),
                    ):
                        assert syntax.main(argv) == 0
                    read.assert_called_once_with(expected)
                    payload = json.loads(urlopen.call_args.args[0].data)
                    if serialization == "raw":
                        assert payload["stop"] == [END_CODE, TASK, CONSOLE, CODE]
                    else:
                        assert "stop" not in payload
                    assert json.loads((output / "summary.json").read_text())["passed"] == 1

    def test_syntax_preserves_completion_boundaries_and_unmodified_source(self, subtests):
        cases = (
            ("raw", {"content": "x = 1", "stopping_word": END_CODE}, "stop", True, True),
            ("raw", {"content": "x = 1", "stopping_word": END_CODE, "truncated": True}, "missing_end_marker", True, False),
            ("raw", {"content": "x = 1", "stopping_word": "<|task|>"}, "missing_end_marker", True, False),
            ("raw", {"content": None, "stopping_word": END_CODE}, "stop", False, False),
            ("raw", {"content": "```python\nx = 1\n```", "stopping_word": END_CODE}, "stop", False, False),
            ("chat", {"choices": [{"message": {"content": "x = 1"}, "finish_reason": "stop"}]}, "stop", True, True),
            ("chat", {"choices": [{"message": {"content": "x = 1"}, "finish_reason": "length"}]}, "length", True, False),
            ("chat", {"choices": [{"message": {}, "finish_reason": "stop"}]}, "stop", False, False),
        )
        record = {
            "id": "example",
            "family": "example",
            "prompt": "Write Python",
            "messages": [{"role": "user", "content": "Write Python"}, {"role": "assistant", "content": "x = 1"}],
        }
        for serialization, completion, finish_reason, compiles, passed in cases:
            with (
                subtests.test(serialization=serialization, completion=completion),
                tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary,
            ):
                root = Path(temporary)
                data = root / "validation.jsonl"
                data.write_text(json.dumps(record) + "\n")
                output = root / "syntax"
                with (
                    patch.object(syntax.urllib.request, "urlopen", return_value=io.StringIO(json.dumps(completion))),
                    redirect_stdout(io.StringIO()),
                ):
                    result = syntax.main(
                        ["--model", "fixture", "--serialization", serialization, "--data", str(data), "--output", str(output)]
                    )
                response = json.loads((output / "responses.jsonl").read_text())
                assert response["finish_reason"] == finish_reason
                assert response["compiles"] == compiles
                assert response["passed"] == passed
                source = completion.get("content") if serialization == "raw" else completion["choices"][0]["message"].get("content")
                assert response["source"] == (source or "")
                assert result == (0 if passed else 1)
