import torch
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from model.benchmark.src import runtime
from pyshell.serialization import CODE, TASK, CONSOLE, END_CODE
from model.benchmark.src.inference import HFCompletion


class Inputs(dict):
    def __init__(self):
        super().__init__(input_ids=torch.tensor([[1, 2]]))
        self.input_ids = self["input_ids"]

    def to(self, device):
        return self


class TestInference:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tokenizer = MagicMock()
        self.tokenizer.return_value = Inputs()
        self.model = MagicMock(device=torch.device("cpu"))
        self.model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model

    def completion(self, *, base_model="fixture-base", adapter=None, device="cpu", revision=None, max_new_tokens=384):
        with (
            patch("model.benchmark.src.runtime.torch.set_num_interop_threads"),
            patch("model.benchmark.src.runtime.torch.set_num_threads"),
            patch("model.benchmark.src.inference.AutoTokenizer.from_pretrained", return_value=self.tokenizer) as tokenizer,
            patch("model.benchmark.src.inference.AutoModelForImageTextToText.from_pretrained", return_value=self.model) as model,
            patch("model.benchmark.src.inference.PeftModel.from_pretrained", return_value=self.model) as load_adapter,
            patch("model.benchmark.src.inference.StopStringCriteria", return_value=SimpleNamespace()) as build,
        ):
            completion = HFCompletion(adapter, 3, device, base_model=base_model, revision=revision, max_new_tokens=max_new_tokens)
            build.assert_called_once_with(self.tokenizer, [END_CODE, TASK, CONSOLE, CODE])
            tokenizer.assert_called_once_with(base_model, revision=revision)
            model.assert_called_once_with(base_model, revision=revision, dtype=torch.float32, attn_implementation="sdpa")
            if adapter is None:
                load_adapter.assert_not_called()
            else:
                load_adapter.assert_called_once_with(self.model, adapter)
        return completion

    def test_recorded_revision_and_configured_generation_budget_are_used(self):
        completion = self.completion(revision="a" * 40, max_new_tokens=77)
        self.tokenizer.decode.return_value = "result = 1" + END_CODE
        completion("prompt")
        assert self.model.generate.call_args.kwargs["max_new_tokens"] == 77

    def test_explicit_base_model_does_not_depend_on_training_artifacts(self, tmp_path):
        self.completion(base_model=tmp_path / "base-model")

    @pytest.mark.parametrize("adapter", [None, "fixture-adapter"])
    def test_base_model_and_adapter_evaluation_remain_supported(self, adapter):
        self.completion(adapter=adapter)

    @pytest.mark.parametrize("available", [False, True])
    def test_auto_device_uses_resolved_device_and_runtime_metadata(self, available):
        with (
            patch("model.benchmark.src.runtime.torch.cuda.is_available", return_value=available),
            patch("model.benchmark.src.inference.torch.cuda.get_device_name", return_value="fixture-gpu") as gpu,
        ):
            completion = self.completion(device="auto")
        self.model.to.assert_called_once_with("cuda" if available else "cpu")
        assert completion.runtime["gpu"] == ("fixture-gpu" if available else None)
        assert completion.runtime["tf32"] is False
        if available:
            gpu.assert_called_once_with()
        else:
            gpu.assert_not_called()

    def test_runtime_failures_propagate_before_model_loading(self, subtests):
        for device, available in (("cuda", False), ("cpu", False), ("cuda", True)):
            with (
                subtests.test(device=device, available=available),
                patch.object(runtime, "torch") as runtime_torch,
                patch("model.benchmark.src.inference.AutoTokenizer.from_pretrained") as tokenizer,
                patch("model.benchmark.src.inference.AutoModelForImageTextToText.from_pretrained") as model,
            ):
                runtime_torch.cuda.is_available.return_value = available
                error = RuntimeError("interop threads already initialized")
                runtime_torch.set_num_interop_threads.side_effect = error
                with pytest.raises(RuntimeError) as raised:
                    HFCompletion(None, 3, device, base_model="fixture-base")
                if device == "cuda" and not available:
                    assert str(raised.value) == "CUDA requested but unavailable"
                    runtime_torch.set_num_threads.assert_not_called()
                    runtime_torch.set_num_interop_threads.assert_not_called()
                else:
                    assert raised.value is error
                    runtime_torch.set_num_threads.assert_called_once_with(3)
                    runtime_torch.set_num_interop_threads.assert_called_once_with(1)
                tokenizer.assert_not_called()
                model.assert_not_called()

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

    def test_earliest_boundary_and_missing_marker_keep_existing_semantics(self, subtests):
        completion = self.completion()
        for text, expected in (
            ("code" + TASK + END_CODE, ("code", False)),
            ("code" + CONSOLE + END_CODE, ("code", False)),
            ("code" + CODE + END_CODE, ("code", False)),
            ("code" + END_CODE + CONSOLE, ("code", True)),
            ("code", ("code", False)),
        ):
            with subtests.test(text=text):
                self.tokenizer.decode.return_value = text
                assert completion("prompt") == expected
