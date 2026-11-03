"""Run the unquantized adapter directly for CPU training/export comparisons."""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from dataset.benchmark.src.serialization import CODE, CONSOLE, END_CODE, TASK

ROOT = Path(__file__).resolve().parents[1]


class HFCompletion:
    def __init__(self, adapter=None, threads=16):
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            ROOT / "artifacts/base", local_files_only=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            ROOT / "artifacts/base",
            dtype=torch.float32,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def __call__(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=384,
                use_cache=True,
                stop_strings=[END_CODE, TASK, CONSOLE, CODE],
                tokenizer=self.tokenizer,
            )
        text = self.tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=False
        )
        boundaries = [
            (text.index(marker), marker)
            for marker in (END_CODE, TASK, CONSOLE, CODE)
            if marker in text
        ]
        if not boundaries:
            return text, False
        offset, marker = min(boundaries)
        return text[:offset], marker == END_CODE
