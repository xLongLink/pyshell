"""LoRA pilot with loss restricted to the final executable response."""

import argparse
import ctypes
import gc
import json
import math
import random
import time
from pathlib import Path

import psutil
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from pyshell.serialization import FORMAT
from transformers import AutoModelForImageTextToText, AutoTokenizer

from model.benchmark.src.corpus import default_training_data, read_jsonl, sha256
from model.benchmark.src.resources import exclusive_cpu
from model.benchmark.src.runtime import configure_device, configure_threads
from model.train.src.checkpoint_evaluation import evaluate_checkpoint
from model.train.src.checkpoints import load_resume, save_checkpoint
from model.train.src.supervision import encode, evaluate_loss, loss_for

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "artifacts/base")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/adapter")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=60)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument("--initialize-adapter", type=Path)
    parser.add_argument("--evaluate-every", type=int, default=0)
    args = parser.parse_args(argv)
    args.data = args.data or default_training_data("raw")
    if args.stop_after is not None and (
        not 0 < args.stop_after <= args.steps or args.stop_after % args.accumulation
    ):
        parser.error(
            "--stop-after must be within --steps and on an accumulation boundary"
        )
    if args.evaluate_every < 0 or args.evaluate_every % args.accumulation:
        parser.error(
            "--evaluate-every must be zero or a positive multiple of gradient accumulation"
        )
    return args


def train(args):
    configure_device(args.device)
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing training artifacts: {args.output}"
        )
    args.output.mkdir(parents=True)
    configure_threads(args.threads)
    torch.manual_seed(713)
    random.seed(713)
    libc = ctypes.CDLL(None)
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
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
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
    updates = math.ceil(args.steps / args.accumulation)
    warmup = min(10, max(1, updates // 10))
    optimizer.zero_grad(set_to_none=True)
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
        "serialization": "raw",
        "serialization_format": FORMAT,
        "evaluate_every": args.evaluate_every,
    }

    try:
        print(
            f"Training with {torch.get_num_threads()} threads, starting at example {resume_step + 1}; optimizer restored: {optimizer_restored}",
            flush=True,
        )
        with (args.output / "steps.jsonl").open("a") as file:
            for step in range(resume_step, stop_step):
                item = train[indices[step % len(indices)]]
                step_started = time.monotonic()
                group_start = (step // args.accumulation) * args.accumulation
                group_size = min(args.accumulation, args.steps - group_start)
                loss = loss_for(model, item)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step + 1}")
                (loss / group_size).backward()
                if (step + 1) % args.accumulation == 0 or step + 1 == args.steps:
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    update = status["updates"] + 1
                    multiplier = (
                        update / warmup
                        if update <= warmup
                        else 0.1
                        + 0.9
                        * (
                            1
                            + math.cos(
                                math.pi * (update - warmup) / max(1, updates - warmup)
                            )
                        )
                        / 2
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = args.learning_rate * multiplier
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    status["updates"] = update
                loss_value = float(loss.detach())
                del loss
                gc.collect()
                if hasattr(libc, "malloc_trim"):
                    libc.malloc_trim(0)
                row = {
                    "step": step + 1,
                    "id": item["id"],
                    "loss": loss_value,
                    "seconds": time.monotonic() - step_started,
                    "rss_gib": psutil.Process().memory_info().rss / 2**30,
                }
                if args.device == "cuda":
                    row["peak_cuda_gib"] = torch.cuda.max_memory_allocated() / 2**30
                status["completed_steps"] = step + 1
                file.write(json.dumps(row) + "\n")
                file.flush()
                if step - resume_step < 3 or (step + 1) % 10 == 0:
                    elapsed = time.monotonic() - started
                    print(
                        f"Step {step + 1}/{args.steps}: loss={row['loss']:.4f}, step={row['seconds']:.2f}s, RAM={row['rss_gib']:.2f}GiB, ETA={(args.steps - step - 1) * elapsed / (step - resume_step + 1) / 60:.1f}min",
                        flush=True,
                    )
                evaluate_now = (
                    args.evaluate_every
                    and (step + 1) % args.evaluate_every == 0
                    and step + 1 < args.steps
                )
                if (step + 1) % 100 == 0 or evaluate_now:
                    status["training_seconds"] = (
                        prior_seconds + time.monotonic() - started
                    )
                    save_checkpoint(
                        model,
                        tokenizer,
                        optimizer,
                        status,
                        args.output,
                        f"checkpoint-{step + 1}",
                    )
                if evaluate_now:
                    print(
                        f"Pausing training for checkpoint {step + 1} evaluation.",
                        flush=True,
                    )
                    status.update(status="evaluating", evaluation_step=step + 1)
                    (args.output / "training.json").write_text(
                        json.dumps(status, indent=2) + "\n"
                    )
                    returncode = evaluate_checkpoint(
                        (args.output / f"checkpoint-{step + 1}").resolve(),
                        step + 1,
                        (args.output / "evaluations.jsonl").resolve(),
                    )
                    print(
                        f"Resuming training; checkpoint evaluation exit code: {returncode}.",
                        flush=True,
                    )
                    status.update(
                        status="running",
                        last_evaluation_step=step + 1,
                        last_evaluation_returncode=returncode,
                    )
                    (args.output / "training.json").write_text(
                        json.dumps(status, indent=2) + "\n"
                    )
        status.update(
            status="completed" if stop_step == args.steps else "paused",
            training_seconds=prior_seconds + time.monotonic() - started,
            validation_loss_after=evaluate_loss(model, validation),
        )
        save_checkpoint(model, tokenizer, optimizer, status, args.output, "final")
        print(json.dumps(status, indent=2), flush=True)
    except BaseException:
        status.update(
            status="interrupted",
            training_seconds=prior_seconds + time.monotonic() - started,
        )
        save_checkpoint(model, tokenizer, optimizer, status, args.output, "interrupted")
        raise


def main(argv=None):
    args = parse_args(argv)
    with exclusive_cpu():
        train(args)
