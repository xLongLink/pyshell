import hashlib
from dataclasses import dataclass
from pathlib import Path

import datasets
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer, HfArgumentParser
from trl import SFTConfig, SFTTrainer

from model.train.src.checkpoints import Checkpoints, load_resume
from model.train.src.data import prepare

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
    if config.completion_only_loss is False:
        raise ValueError("PyShell requires completion-only loss")
    data = Path(inputs.data)
    if not (data / "train.jsonl").stat().st_size:
        raise ValueError("Training dataset must not be empty")
    validation = data / "validation.jsonl"
    files = {"train": str(data / "train.jsonl")}
    if validation.is_file() and validation.stat().st_size:
        files["validation"] = str(validation)
    tokenizer = AutoTokenizer.from_pretrained(
        inputs.base,
        revision=(config.model_init_kwargs or {}).get("revision"),
        trust_remote_code=config.trust_remote_code,
    )
    if config.eos_token is not None:
        tokenizer.eos_token = config.eos_token
    metadata = {
        "base_model": str(Path(inputs.base).resolve())
        if Path(inputs.base).exists()
        else inputs.base,
        "base_revision": (config.model_init_kwargs or {}).get("revision"),
        "accumulation": config.gradient_accumulation_steps,
        "max_length": config.max_length,
        "eos_token": tokenizer.eos_token,
        "serialization": "raw-completion-eos-v1",
    }
    for split, filename in files.items():
        with Path(filename).open("rb") as stream:
            metadata[f"{split}_sha256"] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    if config.resume_from_checkpoint:
        completed = load_resume(Path(config.resume_from_checkpoint), metadata)
        if 0 < config.max_steps <= completed:
            raise ValueError("Training budget must follow the resumed checkpoint")
    dataset = datasets.load_dataset("json", data_files=files)
    dataset = dataset.select_columns(["prompt", "completion"]).map(
        prepare,
        fn_kwargs={"tokenizer": tokenizer, "max_length": config.max_length},
    )
    lora = (
        LoraConfig.from_pretrained(config.resume_from_checkpoint)
        if config.resume_from_checkpoint
        else LoraConfig(
            r=8,
            lora_alpha=16,
            task_type="CAUSAL_LM",
            target_modules=r".*language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|out_proj))",
        )
    )
    lora.inference_mode = False
    model = inputs.base
    if inputs.initialize_adapter:
        model_kwargs = dict(config.model_init_kwargs or {})
        model_kwargs.setdefault("trust_remote_code", config.trust_remote_code)
        model = AutoModelForImageTextToText.from_pretrained(inputs.base, **model_kwargs)
        model = PeftModel.from_pretrained(
            model, inputs.initialize_adapter, is_trainable=True
        )
    checkpoints = Checkpoints(metadata)
    trainer = SFTTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=None if inputs.initialize_adapter else lora,
        callbacks=[checkpoints],
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()
    checkpoints.write(output)


def main(argv=None):
    inputs, config = parse_args(argv)
    train(inputs, config)
