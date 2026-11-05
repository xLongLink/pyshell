from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import datasets
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer, HfArgumentParser
from trl import SFTConfig, SFTTrainer

from model.benchmark.src.corpus import sha256
from model.benchmark.src.resources import exclusive_cpu
from model.train.src.checkpoints import Checkpoints, load_resume
from model.train.src.supervision import encode

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Inputs:
    base: str = str(ROOT / "artifacts/base")
    data: str = str(ROOT / "data/raw")
    initialize_adapter: str | None = None


def parse_args(argv=None):
    parser = HfArgumentParser((Inputs, SFTConfig))
    return parser.parse_args_into_dataclasses(argv)


def train(inputs, config):
    output = Path(config.output_dir)
    if output.exists() and not config.resume_from_checkpoint:
        raise FileExistsError(
            f"Refusing to overwrite existing training artifacts: {output}"
        )
    if inputs.initialize_adapter and config.resume_from_checkpoint:
        raise ValueError(
            "Adapter initialization and checkpoint resume cannot be combined"
        )
    data = Path(inputs.data)
    if not (data / "train.jsonl").stat().st_size:
        raise ValueError("Training dataset must not be empty")
    validation = data / "validation.jsonl"
    files = {"train": str(data / "train.jsonl")}
    if validation.is_file() and validation.stat().st_size:
        files["validation"] = str(validation)
    metadata = {
        "base_model": str(Path(inputs.base).resolve())
        if Path(inputs.base).exists()
        else inputs.base,
        "base_revision": (config.model_init_kwargs or {}).get("revision"),
        "train_sha256": sha256(data / "train.jsonl"),
        "validation_sha256": sha256(validation) if validation.is_file() else None,
        "accumulation": config.gradient_accumulation_steps,
        "max_length": config.max_length,
        "serialization": "raw",
    }
    if config.resume_from_checkpoint:
        completed = load_resume(Path(config.resume_from_checkpoint), metadata)
        if 0 < config.max_steps <= completed:
            raise ValueError("Training budget must follow the resumed checkpoint")
    with exclusive_cpu() if config.device.type == "cpu" else nullcontext():
        tokenizer = AutoTokenizer.from_pretrained(
            inputs.base,
            revision=metadata["base_revision"],
            trust_remote_code=config.trust_remote_code,
        )
        dataset = datasets.load_dataset("json", data_files=files)
        dataset = dataset.map(
            encode,
            fn_kwargs={"tokenizer": tokenizer, "max_length": config.max_length},
            remove_columns=["prompt", "completion"],
        )
        model = inputs.base
        adapter = config.resume_from_checkpoint or inputs.initialize_adapter
        if adapter:
            model_kwargs = dict(config.model_init_kwargs or {})
            model_kwargs.setdefault("trust_remote_code", config.trust_remote_code)
            model = AutoModelForImageTextToText.from_pretrained(
                inputs.base, **model_kwargs
            )
            model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            task_type="CAUSAL_LM",
            target_modules=r".*language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|out_proj))",
        )
        checkpoints = Checkpoints(metadata)
        trainer = SFTTrainer(
            model=model,
            args=config,
            processing_class=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            peft_config=None if adapter else lora,
            callbacks=[checkpoints],
        )
        trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
        trainer.save_model()
        trainer.save_state()
        checkpoints.write(output)


def main(argv=None):
    inputs, config = parse_args(argv)
    train(inputs, config)
