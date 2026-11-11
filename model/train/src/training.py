import math
from pathlib import Path

import datasets
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from model.train.src import settings
from model.train.src.data import prepare


def train(config: settings.Settings):
    if config.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing training artifacts: {config.output}"
        )
    data = Path(config.data)
    if not (data / "train.jsonl").stat().st_size:
        raise ValueError("Training dataset must not be empty")
    validation = data / "validation.jsonl"
    files = {"train": str(data / "train.jsonl")}
    if validation.is_file() and validation.stat().st_size:
        files["validation"] = str(validation)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base, revision=config.base_revision
    )
    dataset = datasets.load_dataset("json", data_files=files)
    dataset = dataset.select_columns(["prompt", "completion"]).map(
        prepare,
        fn_kwargs={"tokenizer": tokenizer, "max_length": config.max_length},
    )
    lora = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        task_type="CAUSAL_LM",
        revision=config.base_revision,
        target_modules=config.target_modules,
    )
    trainer_args = SFTConfig(
        output_dir=str(config.output),
        seed=config.seed,
        bf16=config.dtype == "bfloat16",
        use_cpu=config.cpu,
        num_train_epochs=config.epochs,
        data_seed=config.seed,
        max_length=config.max_length,
        save_strategy="no",
        eval_strategy="epoch" if "validation" in dataset else "no",
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
    trainer = SFTTrainer(
        model=config.base,
        args=trainer_args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=lora,
    )
    batch_size = trainer.args.train_batch_size * trainer.args.world_size
    effective_batch_size = batch_size * config.gradient_accumulation_steps
    batches = math.ceil(len(dataset["train"]) / batch_size)
    steps = math.ceil(batches / config.gradient_accumulation_steps) * config.epochs
    print(
        f"Training: {len(dataset['train'])} rows, {config.epochs} epoch(s), "
        f"effective batch size {effective_batch_size}, {steps} optimizer steps",
        flush=True,
    )
    trainer.train()
    trainer.save_model()
    trainer.save_state()
