import hashlib
from pathlib import Path

import datasets
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from model.train.src import settings
from model.train.src.checkpoints import Checkpoints, load_resume
from model.train.src.data import prepare


def train(config: settings.Settings):
    if config.output.exists() and config.resume is None:
        raise FileExistsError(
            f"Refusing to overwrite existing training artifacts: {config.output}"
        )
    data = Path(config.data)
    data_revision = None
    if not data.is_dir():
        snapshot = snapshot_download(
            config.data,
            repo_type="dataset",
            revision=config.data_revision,
            allow_patterns=["data/train.jsonl", "data/validation.jsonl"],
        )
        data = Path(snapshot) / "data"
        data_revision = Path(snapshot).name
    if not (data / "train.jsonl").stat().st_size:
        raise ValueError("Training dataset must not be empty")
    validation = data / "validation.jsonl"
    files = {"train": str(data / "train.jsonl")}
    if validation.is_file() and validation.stat().st_size:
        files["validation"] = str(validation)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base, revision=config.base_revision
    )
    metadata = {
        "base_model": str(Path(config.base).resolve())
        if Path(config.base).exists()
        else config.base,
        "base_revision": config.base_revision,
        "data_revision": data_revision,
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
    if config.resume:
        completed = load_resume(config.resume, metadata)
        if config.steps <= completed:
            raise ValueError("Training budget must follow the resumed checkpoint")
    dataset = datasets.load_dataset("json", data_files=files)
    dataset = dataset.select_columns(["prompt", "completion"]).map(
        prepare,
        fn_kwargs={"tokenizer": tokenizer, "max_length": config.max_length},
    )
    lora = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        task_type="CAUSAL_LM",
        target_modules=config.target_modules
        if isinstance(config.target_modules, str)
        else list(config.target_modules),
    )
    adapter = config.resume or config.initialize_adapter
    if adapter:
        saved_lora = LoraConfig.from_pretrained(adapter)
        if any(
            getattr(lora, field) != getattr(saved_lora, field)
            for field in ("r", "lora_alpha", "target_modules")
        ):
            raise ValueError(
                "Saved adapter LoRA settings differ; match its settings or train a fresh adapter"
            )
        lora = saved_lora
    lora.inference_mode = False
    trainer_args = SFTConfig(
        output_dir=str(config.output),
        seed=config.seed,
        bf16=config.dtype == "bfloat16",
        use_cpu=config.cpu,
        max_steps=config.steps,
        data_seed=config.seed,
        max_length=config.max_length,
        save_steps=config.save_steps,
        eval_steps=config.save_steps,
        eval_strategy="steps" if "validation" in dataset else "no",
        eval_on_start="validation" in dataset,
        logging_steps=1,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={
            "revision": config.base_revision,
            "dtype": config.dtype,
            "attn_implementation": "sdpa",
        },
        completion_only_loss=True,
        optim="adamw_torch",
        report_to="none",
    )
    model = config.base
    if config.initialize_adapter:
        model = AutoModelForImageTextToText.from_pretrained(
            config.base, **trainer_args.model_init_kwargs
        )
        model = PeftModel.from_pretrained(
            model, config.initialize_adapter, is_trainable=True
        )
    checkpoints = Checkpoints(metadata, config)
    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=None if config.initialize_adapter else lora,
        callbacks=[checkpoints],
    )
    trainer.train(resume_from_checkpoint=str(config.resume) if config.resume else None)
    trainer.save_model()
    trainer.save_state()
    checkpoints.write(config.output)


def main():
    train(settings.SETTINGS)
