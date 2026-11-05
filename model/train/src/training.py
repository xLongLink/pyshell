import argparse
import math
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from model.benchmark.src.corpus import read_jsonl, sha256
from model.benchmark.src.resources import exclusive_cpu
from model.benchmark.src.runtime import configure_device, configure_threads
from model.train.src.checkpoints import Checkpoints, load_resume
from model.train.src.supervision import encode

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "artifacts/base")
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/adapter")
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Example budget, rounded up to a complete accumulation group",
    )
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
    if min(args.steps, args.accumulation, args.max_length, args.threads) < 1:
        parser.error(
            "--steps, --accumulation, --max-length and --threads must be positive"
        )
    if args.eval_samples < 0:
        parser.error("--eval-samples must not be negative")
    if args.stop_after is not None and (
        not 0 < args.stop_after <= args.steps or args.stop_after % args.accumulation
    ):
        parser.error(
            "--stop-after must be within --steps and on an accumulation boundary"
        )
    return args


def validation_dataset(records, tokenizer, max_length, eval_samples):
    if not records or not eval_samples:
        return None
    selected = (
        Dataset.from_dict({"index": range(len(records))})
        .shuffle(seed=713)
        .select(range(min(eval_samples, len(records))))
    )
    return Dataset.from_list(
        [encode(records[index], tokenizer, max_length) for index in selected["index"]]
    )


def train(args):
    args.device = configure_device(args.device)
    if (
        any(
            int(os.environ.get(name, "1")) != 1
            for name in (
                "WORLD_SIZE",
                "PMI_SIZE",
                "OMPI_COMM_WORLD_SIZE",
                "MV2_COMM_WORLD_SIZE",
            )
        )
        or (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() != 1
        )
        or (args.device == "cuda" and torch.cuda.device_count() > 1)
    ):
        raise ValueError("Training requires one process and one CPU or CUDA device")
    if os.environ.get("ACCELERATE_MIXED_PRECISION", "no").lower() != "no":
        raise ValueError("Training requires float32 without Accelerate mixed precision")
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing training artifacts: {args.output}"
        )
    with exclusive_cpu() if args.device == "cpu" else nullcontext():
        args.output.mkdir(parents=True)
        configure_threads(args.threads)
        set_seed(713)
        metadata = {
            "base_model": str(args.base.resolve()),
            "train_sha256": sha256(args.data / "train.jsonl"),
            "validation_sha256": sha256(args.data / "validation.jsonl"),
            "accumulation": args.accumulation,
            "max_length": args.max_length,
            "serialization": "raw",
        }
        updates = math.ceil(args.steps / args.accumulation)
        stop_update = (
            args.stop_after // args.accumulation if args.stop_after else updates
        )
        if args.resume:
            completed = load_resume(args.resume, metadata)
            if stop_update <= completed:
                raise ValueError("Stopping step must follow the resumed checkpoint")

        tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
        dataset = Dataset.from_list(
            [
                encode(record, tokenizer, args.max_length)
                for record in read_jsonl(args.data / "train.jsonl")
            ]
        )
        if not len(dataset):
            raise ValueError("Training dataset must not be empty")
        evaluation = validation_dataset(
            read_jsonl(args.data / "validation.jsonl"),
            tokenizer,
            args.max_length,
            args.eval_samples,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            args.base,
            dtype=torch.float32,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=r".*language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|out_proj))",
        )
        adapter = args.resume or args.initialize_adapter
        if adapter:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        config = SFTConfig(
            output_dir=str(args.output),
            use_cpu=args.device == "cpu",
            bf16=False,
            fp16=False,
            tf32=False,
            seed=713,
            data_seed=713,
            max_steps=updates,
            max_length=None,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.accumulation,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine_with_min_lr",
            lr_scheduler_kwargs={"min_lr_rate": 0.1},
            warmup_steps=min(10, max(1, updates // 10)),
            optim="adamw_torch",
            adam_beta2=0.95,
            weight_decay=0.01,
            completion_only_loss=True,
            prediction_loss_only=True,
            eval_on_start=evaluation is not None,
            eval_strategy="steps" if evaluation is not None else "no",
            eval_steps=updates,
            save_steps=max(1, math.ceil(100 / args.accumulation)),
            logging_steps=1,
            report_to="none",
            dataloader_pin_memory=args.device == "cuda",
        )
        checkpoints = Checkpoints(metadata, stop_update)
        trainer = SFTTrainer(
            model=model,
            args=config,
            processing_class=tokenizer,
            train_dataset=dataset,
            eval_dataset=evaluation,
            peft_config=None if adapter else lora,
            callbacks=[checkpoints],
        )
        trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
        trainer.save_model()
        trainer.save_state()
        checkpoints.write(args.output)


def main(argv=None):
    args = parse_args(argv)
    train(args)
