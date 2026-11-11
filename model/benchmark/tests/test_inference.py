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
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        self.adapter = tmp_path / "adapter"
        self.adapter.mkdir()
        self.tokenizer = MagicMock()
        self.tokenizer.return_value = BatchEncoding({"input_ids": torch.tensor([[1, 2]])})
        self.model = MagicMock(device=torch.device("cpu"))
        self.model.generation_config.eos_token_id = 99
        self.model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        self.model.to.return_value = self.model

        with (
            patch("model.benchmark.src.inference.torch.set_num_threads") as self.threads,
            patch("model.benchmark.src.inference.AutoTokenizer.from_pretrained", return_value=self.tokenizer) as self.load_tokenizer,
            patch("model.benchmark.src.inference.AutoModelForImageTextToText.from_pretrained", return_value=self.model) as self.load_model,
            patch("model.benchmark.src.inference.PeftModel.from_pretrained", return_value=self.model) as self.load_adapter,
            patch("model.benchmark.src.inference.StopStringCriteria", return_value=SimpleNamespace()) as self.build,
        ):
            try:
                yield
            finally:
                torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
                torch.backends.cudnn.allow_tf32 = cudnn_tf32

    def completion(self, *, device="cpu", revision=None, max_new_tokens=384):
        (self.adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "fixture-base", "revision": revision}))
        config = Settings(adapter=self.adapter, threads=3, device=device, tokens=max_new_tokens)
        completion = HFCompletion(config)
        return completion  # noqa: RET504

    @pytest.mark.parametrize("revision", [None, "a" * 40])
    def test_adapter_loads_its_recorded_base_model(self, revision):
        self.completion(revision=revision)
        self.threads.assert_called_once_with(3)
        self.load_tokenizer.assert_called_once_with("fixture-base", revision=revision)
        self.load_model.assert_called_once_with("fixture-base", revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa")
        self.load_adapter.assert_called_once_with(self.model, self.adapter)

    def test_configured_generation_budget_is_used(self):
        completion = self.completion(max_new_tokens=77)
        self.tokenizer.decode.return_value = "result = 1" + END_CODE
        completion(["prompt"])
        assert self.model.generate.call_args.kwargs["max_new_tokens"] == 77

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_configured_device(self, device):
        self.completion(device=device)
        self.model.to.assert_called_once_with(device)
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32

    def test_boundary_matcher_is_built_once_and_reused_across_requests(self):
        completion = self.completion()
        self.tokenizer.decode.return_value = "result = 1\n" + END_CODE
        for _ in range(2):
            assert completion(["prompt"]) == [("result = 1\n", True)]
            options = self.model.generate.call_args.kwargs
            assert options["stopping_criteria"] is completion.stopping_criteria
            assert options["stop_strings"] is None
            assert not options["do_sample"]
        self.build.assert_called_once_with(self.tokenizer, [END_CODE, TASK, CONSOLE, CODE])

    def test_decoded_completion_is_parsed(self):
        completion = self.completion()
        self.tokenizer.decode.return_value = "code" + TASK + END_CODE
        assert completion(["prompt"]) == [("code", False)]

    def test_batch_decoding_removes_padding_after_each_rows_eos(self):
        completion = self.completion()
        self.model.generation_config.eos_token_id = [98, 99]
        self.model.generate.return_value = torch.tensor([[1, 2, 3, 99, 99], [1, 2, 4, 5, 98]])
        self.tokenizer.decode.side_effect = ["result = 1" + END_CODE, "result = 2" + END_CODE]
        assert completion(["short", "long"]) == [("result = 1", True), ("result = 2", True)]
        assert [call.args[0] for call in self.tokenizer.decode.call_args_list] == [[3, 99], [4, 5, 98]]
        assert self.tokenizer.padding_side == "left"


@pytest.mark.parametrize("failure", [False, True])
def test_inference_fixture_restores_tf32_settings(tmp_path, failure):
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    fixture = TestInference()
    setup = TestInference.setup.__wrapped__(fixture, tmp_path)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        next(setup)
        fixture.completion()
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        if failure:
            with pytest.raises(RuntimeError, match="test failed"):
                setup.throw(RuntimeError("test failed"))
        else:
            with pytest.raises(StopIteration):
                next(setup)
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
    finally:
        setup.close()
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
