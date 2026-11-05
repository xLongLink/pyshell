import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import psutil
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from pyshell.serialization import FORMAT
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from model.benchmark.src.corpus import read_jsonl, sha256
from model.benchmark.src.resources import exclusive_cpu
from model.benchmark.src.runtime import configure_device, configure_threads
from model.train.src.checkpoints import load_resume, save_checkpoint
from model.train.src.supervision import encode, evaluate_loss, loss_for

ROOT = Path(__file__).resolve().parents[1]


def scheduler_for(optimizer, steps, accumulation, completed_updates=0):
    updates = math.ceil(steps / accumulation)
    warmup = min(10, max(1, updates // 10))

    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return get_cosine_with_min_lr_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=updates,
        min_lr_rate=0.1,
        last_epoch=completed_updates,
    )


def single_example(items):
    return items[0]


class ResponseTrainer(Trainer):
    def __init__(self, *args, status, log, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.world_size != 1 or self.args.n_gpu > 1:
            raise ValueError("Training requires one process and one CPU or CUDA device")
        if self.accelerator.mixed_precision != "no":
            raise ValueError(
                "Training requires float32 without Accelerate mixed precision"
            )
        self.status = status
        self.step_log = log
        self.model_accepts_loss_kwargs = False

    def _get_train_sampler(self, train_dataset=None):
        return torch.utils.data.SequentialSampler(
            self.train_dataset if train_dataset is None else train_dataset
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss = loss_for(model, inputs)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Nonfinite loss at step {self.status['completed_steps'] + 1}"
            )
        self.response_loss = float(loss.detach())
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        started = time.monotonic()
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.status["completed_steps"] += 1
        row = {
            "step": self.status["completed_steps"],
            "id": inputs["id"],
            "loss": self.response_loss,
            "seconds": time.monotonic() - started,
            "rss_gib": psutil.Process().memory_info().rss / 2**30,
        }
        if self.args.device.type == "cuda":
            row["peak_cuda_gib"] = torch.cuda.max_memory_allocated() / 2**30
        self.step_log.write(json.dumps(row) + "\n")
        self.step_log.flush()
        return loss


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "artifacts/base")
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/adapter")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=60)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument("--initialize-adapter", type=Path)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.accumulation < 1:
        parser.error("--steps and --accumulation must be positive")
    if args.stop_after is not None and (
        not 0 < args.stop_after <= args.steps or args.stop_after % args.accumulation
    ):
        parser.error(
            "--stop-after must be within --steps and on an accumulation boundary"
        )
    return args


def train(args):
    args.device = configure_device(args.device)
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing training artifacts: {args.output}"
        )
    args.output.mkdir(parents=True)
    configure_threads(args.threads)
    torch.manual_seed(713)
    random.seed(713)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    train = [
        encode(record, tokenizer, args.max_length)
        for record in read_jsonl(args.data / "train.jsonl")
    ]
    validation = [
        encode(record, tokenizer, args.max_length)
        for record in read_jsonl(args.data / "validation.jsonl")
    ]
    validation = random.Random(713).sample(
        validation, min(args.eval_samples, len(validation))
    )
    print(
        f"Encoded {len(train)} training and {len(validation)} validation examples; lengths {min(len(i['ids']) for i in train)}–{max(len(i['ids']) for i in train)} tokens",
        flush=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.base,
        dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    resume_state = (
        load_resume(args.resume, args.data, args.accumulation) if args.resume else None
    )
    model = (
        PeftModel.from_pretrained(
            model, args.resume or args.initialize_adapter, is_trainable=True
        )
        if args.resume or args.initialize_adapter
        else get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=r".*language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|out_proj))",
            ),
        )
    )
    model.to(args.device)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.print_trainable_parameters()
    baseline = (
        resume_state["validation_loss_before"]
        if resume_state
        else evaluate_loss(model, validation)
    )
    print(f"Held-out loss before training: {baseline}", flush=True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    optimizer_restored = False
    if args.resume and (args.resume / "optimizer.pt").is_file():
        optimizer.load_state_dict(
            torch.load(
                args.resume / "optimizer.pt", weights_only=True, map_location="cpu"
            )
        )
        optimizer_restored = True
    indices = list(range(len(train)))
    random.Random(713).shuffle(indices)
    for group in optimizer.param_groups:
        group["initial_lr"] = args.learning_rate
    scheduler = scheduler_for(
        optimizer,
        args.steps,
        args.accumulation,
        resume_state["updates"] if resume_state else 0,
    )
    scheduler_restored = False
    if (
        args.resume
        and (args.resume / "scheduler.pt").is_file()
        and resume_state["steps"] == args.steps
        and resume_state["learning_rate"] == args.learning_rate
    ):
        scheduler.load_state_dict(
            torch.load(
                args.resume / "scheduler.pt", weights_only=True, map_location="cpu"
            )
        )
        for group, rate in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group["lr"] = rate
        scheduler_restored = True
    started = time.monotonic()
    resume_step = (
        min(resume_state["updates"] * args.accumulation, args.steps)
        if resume_state
        else 0
    )
    prior_seconds = resume_state.get("training_seconds", 0) if resume_state else 0
    stop_step = args.stop_after or args.steps
    if stop_step <= resume_step:
        raise ValueError("Stopping step must follow the resumed checkpoint")
    status = {
        "status": "running",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "base_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "device": args.device,
        "dtype": "float32",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "tf32": False,
        "rank": model.peft_config["default"].r,
        "learning_rate": args.learning_rate,
        "steps": args.steps,
        "stop_after": args.stop_after,
        "accumulation": args.accumulation,
        "validation_loss_before": baseline,
        "train_sha256": sha256(args.data / "train.jsonl"),
        "updates": resume_state["updates"] if resume_state else 0,
        "completed_steps": resume_step,
        "threads": args.threads,
        "resumed_from": str(args.resume) if args.resume else None,
        "initialized_from": str(args.initialize_adapter)
        if args.initialize_adapter
        else None,
        "resumed_at_step": resume_step,
        "optimizer_restored": optimizer_restored,
        "scheduler_restored": scheduler_restored,
        "serialization": "raw",
        "serialization_format": FORMAT,
    }

    def checkpoint(name):
        status["training_seconds"] = prior_seconds + time.monotonic() - started
        save_checkpoint(
            model, tokenizer, optimizer, status, args.output, name, scheduler=scheduler
        )

    class CheckpointCallback(TrainerCallback):
        def on_substep_end(self, args, state, control, **kwargs):
            if status["completed_steps"] % 100 == 0:
                checkpoint(f"checkpoint-{status['completed_steps']}")

        def on_step_end(self, args, state, control, **kwargs):
            status["updates"] += 1
            self.on_substep_end(args, state, control, **kwargs)

    try:
        print(
            f"Training with {torch.get_num_threads()} threads, starting at example {resume_step + 1}; optimizer restored: {optimizer_restored}",
            flush=True,
        )
        with (args.output / "steps.jsonl").open("a") as file:
            trainer = ResponseTrainer(
                model=model,
                args=TrainingArguments(
                    output_dir=str(args.output),
                    use_cpu=args.device == "cpu",
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=args.accumulation,
                    max_steps=math.ceil((stop_step - resume_step) / args.accumulation),
                    seed=713,
                    data_seed=713,
                    fp16=False,
                    bf16=False,
                    tf32=False if args.device == "cuda" else None,
                    remove_unused_columns=False,
                    dataloader_pin_memory=args.device == "cuda",
                    save_strategy="no",
                    logging_strategy="no",
                    report_to="none",
                    push_to_hub=False,
                    disable_tqdm=True,
                ),
                train_dataset=torch.utils.data.Subset(
                    train,
                    [
                        indices[step % len(indices)]
                        for step in range(resume_step, stop_step)
                    ],
                ),
                data_collator=single_example,
                optimizers=(optimizer, scheduler),
                callbacks=[CheckpointCallback()],
                status=status,
                log=file,
            )
            trainer.train()
        status.update(
            status="completed" if stop_step == args.steps else "paused",
            validation_loss_after=evaluate_loss(model, validation),
        )
        checkpoint("final")
        print(json.dumps(status, indent=2), flush=True)
    except BaseException:
        status["status"] = "interrupted"
        checkpoint("interrupted")
        raise


def main(argv=None):
    args = parse_args(argv)
    args.device = configure_device(args.device)
    with exclusive_cpu() if args.device == "cpu" else nullcontext():
        train(args)
