import json
import torch
import pytest
from types import SimpleNamespace
from transformers import BatchEncoding
from unittest.mock import MagicMock, patch
from pyshell.serialization import CODE, TASK, CONSOLE, END_CODE
from model.benchmark.src.settings import Settings
from model.benchmark.src.inference import HFCompletion


class TestInference:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.adapter = tmp_path / "adapter"
        self.adapter.mkdir()
        self.tokenizer = MagicMock()
        self.tokenizer.return_value = BatchEncoding({"input_ids": torch.tensor([[1, 2]])})
        self.model = MagicMock(device=torch.device("cpu"))
        self.model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model

        with (
            patch("model.benchmark.src.inference.torch.set_num_threads") as self.threads,
            patch("model.benchmark.src.inference.AutoTokenizer.from_pretrained", return_value=self.tokenizer) as self.load_tokenizer,
            patch("model.benchmark.src.inference.AutoModelForImageTextToText.from_pretrained", return_value=self.model) as self.load_model,
            patch("model.benchmark.src.inference.PeftModel.from_pretrained", return_value=self.model) as self.load_adapter,
            patch("model.benchmark.src.inference.StopStringCriteria", return_value=SimpleNamespace()) as self.build,
        ):
            yield

    def completion(self, *, base_model="fixture-base", device="cpu", revision=None, max_new_tokens=384):
        (self.adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base_model), "revision": revision}))
        config = Settings(adapter=self.adapter, threads=3, device=device, tokens=max_new_tokens)
        completion = HFCompletion(config)
        return completion

    @pytest.mark.parametrize("revision", [None, "a" * 40])
    @pytest.mark.parametrize("local", [False, True])
    def test_adapter_loads_its_recorded_base_model(self, tmp_path, revision, local):
        base = tmp_path / "base-model" if local else "fixture-base"
        self.completion(base_model=base, revision=revision)
        self.threads.assert_called_once_with(3)
        self.load_tokenizer.assert_called_once_with(str(base), revision=revision)
        self.load_model.assert_called_once_with(str(base), revision=revision, dtype=torch.float32, attn_implementation="sdpa")
        self.load_adapter.assert_called_once_with(self.model, self.adapter)

    def test_configured_generation_budget_is_used(self):
        completion = self.completion(max_new_tokens=77)
        self.tokenizer.decode.return_value = "result = 1" + END_CODE
        completion("prompt")
        assert self.model.generate.call_args.kwargs["max_new_tokens"] == 77

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_configured_device(self, device):
        self.completion(device=device)
        self.model.to.assert_called_once_with(device)
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32

    def test_device_errors_propagate(self):
        error = RuntimeError("CUDA unavailable")
        self.model.to.side_effect = error
        with pytest.raises(RuntimeError) as raised:
            self.completion(device="cuda")
        assert raised.value is error

    def test_boundary_matcher_is_built_once_and_reused_across_requests(self):
        completion = self.completion()
        self.tokenizer.decode.return_value = "result = 1\n" + END_CODE
        for _ in range(2):
            assert completion("prompt") == ("result = 1\n", True)
            options = self.model.generate.call_args.kwargs
            assert options["stopping_criteria"] is completion.stopping_criteria
            assert options["stop_strings"] is None
            assert "streamer" not in options
            assert options["max_new_tokens"] == 384
            assert not options["do_sample"]
        self.build.assert_called_once_with(self.tokenizer, [END_CODE, TASK, CONSOLE, CODE])

    def test_decoded_completion_is_parsed(self):
        completion = self.completion()
        self.tokenizer.decode.return_value = "code" + TASK + END_CODE
        assert completion("prompt") == ("code", False)
