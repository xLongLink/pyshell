import torch
from peft import PeftModel
from transformers import AutoTokenizer, StopStringCriteria, StoppingCriteriaList, AutoModelForImageTextToText
from pyshell.parsing import parse_completion
from pyshell.serialization import MARKERS
from model.benchmark.src.runtime import configure_device, configure_threads


class HFCompletion:
    def __init__(self, adapter, threads, device, *, base_model, revision=None, max_new_tokens=384):
        device = configure_device(device)
        configure_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
        self.model = AutoModelForImageTextToText.from_pretrained(
            base_model, revision=revision, dtype=torch.float32, attn_implementation="sdpa"
        )
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.to(device).eval()
        self.max_new_tokens = max_new_tokens
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
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stop_strings=None,
                stopping_criteria=self.stopping_criteria,
            )
        text = self.tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=False)
        return parse_completion(text)
