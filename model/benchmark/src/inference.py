import json
import torch
from peft import PeftModel
from transformers import AutoTokenizer, StopStringCriteria, StoppingCriteriaList, AutoModelForImageTextToText
from pyshell.parsing import parse_completion
from pyshell.serialization import MARKERS
from model.benchmark.src.settings import Settings


class HFCompletion:
    """Generate code completions from a trained adapter."""

    def __init__(self, config: Settings) -> None:
        """Load the adapter and configure deterministic generation."""
        native = json.loads((config.adapter / "adapter_config.json").read_text())
        base = native["base_model_name_or_path"]
        revision = native.get("revision")
        torch.set_num_threads(config.threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
        self.model = AutoModelForImageTextToText.from_pretrained(base, revision=revision, dtype=torch.float32, attn_implementation="sdpa")
        self.model = PeftModel.from_pretrained(self.model, config.adapter)
        self.model.to(config.device).eval()
        self.max_new_tokens = config.tokens
        self.stopping_criteria = StoppingCriteriaList([StopStringCriteria(self.tokenizer, list(MARKERS))])

    def __call__(self, prompt: str) -> tuple[str, bool]:
        """Return generated code and whether its closing marker was reached."""
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
