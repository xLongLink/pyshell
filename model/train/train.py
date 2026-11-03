"""CPU LoRA pilot with loss restricted to the final executable response."""

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer

from dataset.benchmark.src.serialization import FORMAT, serialize, target_text

ROOT = Path(__file__).resolve().parent


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def encode(record, tokenizer, max_length, serialization="chat"):
    prefix = (
        (record["prompt"] if "prompt" in record else serialize(record["messages"][:-1]))
        if serialization == "raw"
        else tokenizer.apply_chat_template(
            record["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    target = (
        record["completion"]
        if serialization == "raw" and "completion" in record
        else record["messages"][-1]["content"]
    )
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    target_ids = tokenizer.encode(
        target_text(target) if serialization == "raw" else target + tokenizer.eos_token,
        add_special_tokens=False,
    )
    ids = prompt_ids + target_ids
    if len(ids) > max_length:
        raise ValueError(
            f"{record['id']} needs {len(ids)} tokens, above {max_length}; refusing to truncate supervision"
        )
    return {
        "id": record["id"],
        "ids": ids,
        "prefix_length": len(prompt_ids),
        "target_tokens": len(target_ids),
    }


def loss_for(model, item):
    ids = torch.tensor([item["ids"]], dtype=torch.long)
    start = item["prefix_length"]
    positions = torch.arange(start - 1, ids.shape[1] - 1)
    output = model(input_ids=ids, use_cache=False, logits_to_keep=positions)
    return F.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]), ids[:, start:].reshape(-1)
    )


def evaluate(model, items):
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for item in items:
            loss = loss_for(model, item)
            total += float(loss) * item["target_tokens"]
            tokens += item["target_tokens"]
    model.train()
    return total / tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "artifacts/base")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/adapter")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=60)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-adapter", type=Path)
    parser.add_argument("--serialization", choices=("chat", "raw"), default="raw")
    parser.add_argument("--evaluate-every", type=int, default=0)
    args = parser.parse_args()
    args.data = args.data or (
        ROOT.parents[1] / "dataset/data"
        if args.serialization == "raw"
        else ROOT / "data"
    )
    if args.resume and args.initialize_adapter:
        parser.error("--resume and --initialize-adapter are mutually exclusive")
    if args.evaluate_every < 0 or args.evaluate_every % args.accumulation:
        parser.error(
            "--evaluate-every must be zero or a positive multiple of gradient accumulation"
        )
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing training artifacts: {args.output}"
        )
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(713)
    random.seed(713)
    libc = ctypes.CDLL(None)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    train = [
        encode(record, tokenizer, args.max_length, args.serialization)
        for record in read_records(args.data / "train.jsonl")
    ]
    validation = [
        encode(record, tokenizer, args.max_length, args.serialization)
        for record in read_records(args.data / "validation.jsonl")
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
    model.config.use_cache = False
    resume_state = None
    if args.resume:
        state_file = args.resume / "training.json"
        if not state_file.is_file():
            state_file = args.resume.parent / "training.json"
        resume_state = json.loads(state_file.read_text())
        data_hash = hashlib.sha256((args.data / "train.jsonl").read_bytes()).hexdigest()
        if (
            resume_state["train_sha256"] != data_hash
            or resume_state["accumulation"] != args.accumulation
        ):
            raise ValueError(
                "Resume dataset or gradient accumulation differs from the checkpoint"
            )
        if resume_state.get("serialization", "chat") != args.serialization:
            raise ValueError(
                "Cannot resume an adapter trained with a different serialization"
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
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.print_trainable_parameters()
    baseline = (
        resume_state["validation_loss_before"]
        if resume_state
        else evaluate(model, validation)
        if validation
        else None
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
            torch.load(args.resume / "optimizer.pt", weights_only=True)
        )
        optimizer_restored = True
    indices = list(range(len(train)))
    random.Random(713).shuffle(indices)
    updates = math.ceil(args.steps / args.accumulation)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    logs = []
    resume_step = (
        min(resume_state["updates"] * args.accumulation, args.steps)
        if resume_state
        else 0
    )
    prior_seconds = resume_state.get("training_seconds", 0) if resume_state else 0
    status = {
        "status": "running",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "base_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "device": "cpu",
        "dtype": "float32",
        "rank": 8,
        "learning_rate": args.learning_rate,
        "steps": args.steps,
        "accumulation": args.accumulation,
        "validation_loss_before": baseline,
        "train_sha256": hashlib.sha256(
            (args.data / "train.jsonl").read_bytes()
        ).hexdigest(),
        "updates": resume_state["updates"] if resume_state else 0,
        "threads": args.threads,
        "resumed_from": str(args.resume) if args.resume else None,
        "initialized_from": str(args.initialize_adapter)
        if args.initialize_adapter
        else None,
        "resumed_at_step": resume_step,
        "optimizer_restored": optimizer_restored,
        "serialization": args.serialization,
        "serialization_format": FORMAT
        if args.serialization == "raw"
        else "pretrained-chat-template",
        "evaluate_every": args.evaluate_every,
    }

    def save(name):
        path = args.output if name == "final" else args.output / name
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        torch.save(optimizer.state_dict(), path / "optimizer.pt")
        (path / "training.json").write_text(json.dumps(status, indent=2) + "\n")
        (args.output / "training.json").write_text(json.dumps(status, indent=2) + "\n")

    try:
        print(
            f"Training with {torch.get_num_threads()} threads, starting at example {resume_step + 1}; optimizer restored: {optimizer_restored}",
            flush=True,
        )
        for step in range(resume_step, args.steps):
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
                warmup = min(10, max(1, updates // 10))
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
            logs.append(row)
            with (args.output / "steps.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
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
                status["completed_steps"] = step + 1
                status["training_seconds"] = prior_seconds + time.monotonic() - started
                save(f"checkpoint-{step + 1}")
            if evaluate_now:
                print(
                    f"Pausing training for checkpoint {step + 1} evaluation.",
                    flush=True,
                )
                status.update(status="evaluating", evaluation_step=step + 1)
                (args.output / "training.json").write_text(
                    json.dumps(status, indent=2) + "\n"
                )
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "dataset.benchmark.checkpoint_eval",
                        "--checkpoint",
                        str((args.output / f"checkpoint-{step + 1}").resolve()),
                        "--step",
                        str(step + 1),
                        "--history",
                        str((args.output / "evaluations.jsonl").resolve()),
                        "--serialization",
                        args.serialization,
                    ],
                    cwd=ROOT.parents[1],
                    start_new_session=True,
                )
                try:
                    returncode = process.wait()
                except BaseException:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGINT)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    raise
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
            status="completed",
            completed_steps=args.steps,
            training_seconds=prior_seconds + time.monotonic() - started,
            validation_loss_after=evaluate(model, validation) if validation else None,
        )
        save("final")
        print(json.dumps(status, indent=2), flush=True)
    except BaseException:
        status.update(
            status="interrupted",
            completed_steps=resume_step + len(logs),
            training_seconds=prior_seconds + time.monotonic() - started,
        )
        save("interrupted")
        raise


if __name__ == "__main__":
    main()
