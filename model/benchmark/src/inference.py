"""Run an unquantized adapter directly for execution evaluation."""

import torch
from peft import PeftModel
from pathlib import Path
from transformers import AutoTokenizer, StopStringCriteria, StoppingCriteriaList, AutoModelForImageTextToText
from pyshell.parsing import parse_completion
from pyshell.serialization import MARKERS
from model.benchmark.src.runtime import configure_device, configure_threads

ROOT = Path(__file__).resolve().parents[1]


class HFCompletion:
    def __init__(self, adapter=None, threads=16, device="cpu", *, base_model=None):
        configure_device(device)
        configure_threads(threads)
        base_model = base_model or ROOT / "local/base"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            base_model, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
        )
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.to(device).eval()
        self.stopping_criteria = StoppingCriteriaList([StopStringCriteria(self.tokenizer, list(MARKERS))])
        self.runtime = {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "tf32": False,
        }

    def __call__(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=384, use_cache=True, stop_strings=None, stopping_criteria=self.stopping_criteria
            )
        text = self.tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=False)
        return parse_completion(text)
