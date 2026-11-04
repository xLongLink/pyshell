"""Compare known and held-out execution, plus unquantized HF versus GGUF."""

import gc
import sys
import json
import time
import torch
import random
import argparse
import subprocess
from peft import PeftModel
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForImageTextToText
from model.train.src.export import export_command
from model.train.src.supervision import encode, loss_for
from dataset.benchmark.src.sampling import balanced_sample
from dataset.benchmark.src.serialization import CODE, TASK, CONSOLE, END_CODE

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT.parents[1] / "model/train"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/diagnostic-audit")
    parser.add_argument("--known-samples", type=int, default=50)
    parser.add_argument("--heldout-samples", type=int, default=50)
    parser.add_argument("--hf-samples", type=int, default=5)
    parser.add_argument("--only", choices=("best600", "stopped"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    data = ROOT.parent / "data"
    train = [json.loads(line) for line in (data / "train.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in (data / "validation.jsonl").read_text().splitlines()]
    indices = list(range(len(train)))
    random.Random(713).shuffle(indices)
    known = balanced_sample([train[index] for index in indices[:600]], args.known_samples)
    heldout = balanced_sample(validation, args.heldout_samples)
    records = [{**record, "audit_split": split} for split, group in [("known", known), ("heldout", heldout)] for record in group]
    tokenizer = AutoTokenizer.from_pretrained(TRAIN_ROOT / "artifacts/base", local_files_only=True)
    for record in records:
        record["token_ids"] = tokenizer.encode(record["prompt"], add_special_tokens=False)
    positions = list(range(len(records)))
    loss_inputs = [
        (position, encode(records[position], tokenizer, 2048, "raw"))
        for position in positions[: args.hf_samples] + positions[len(known) : len(known) + args.hf_samples]
    ]
    checkpoints = {
        "best600": TRAIN_ROOT / "artifacts/adapter-raw5000/checkpoint-600",
        "stopped": TRAIN_ROOT / "artifacts/adapter-raw5000/interrupted",
    }
    if args.only:
        checkpoints = {args.only: checkpoints[args.only]}
    for label, checkpoint in checkpoints.items():
        print(f"Auditing {label}: {checkpoint}", flush=True)
        selected = [record.copy() for record in records]
        base = AutoModelForImageTextToText.from_pretrained(
            TRAIN_ROOT / "artifacts/base", dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
        )
        model = PeftModel.from_pretrained(base, checkpoint).eval()
        for position, item in loss_inputs:
            with torch.no_grad():
                selected[position]["hf_target_loss"] = float(loss_for(model, item))
        for record in selected[: args.hf_samples]:
            inputs = tokenizer(record["prompt"], return_tensors="pt", add_special_tokens=False)
            started = time.monotonic()
            with torch.no_grad():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=192, stop_strings=[END_CODE], tokenizer=tokenizer)
            text = tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=False)
            ended = END_CODE in text
            source = text.split(END_CODE, 1)[0] if ended else text
            for marker in (CODE, TASK, CONSOLE):
                if marker in source:
                    ended = False
            record["hf_prediction"] = {"source": source, "end_marker_seen": ended, "seconds": time.monotonic() - started}
            print(f"HF {record['id']}: boundary={ended}, {record['hf_prediction']['seconds']:.2f}s", flush=True)
        del model, base
        gc.collect()
        selection = args.output / f"{label}-examples.jsonl"
        selection.write_text("".join(json.dumps(record) + "\n" for record in selected))
        gguf = checkpoint / "adapter.gguf"
        if not gguf.exists():
            with (args.output / f"{label}-export.log").open("w") as log:
                subprocess.run(export_command(checkpoint, gguf), stdout=log, stderr=subprocess.STDOUT, check=True)
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "dataset.benchmark.main",
                "evaluate",
                "--adapter",
                str(gguf),
                "--serialization",
                "raw",
                "--temperature",
                "0",
                "--label",
                label,
                "--audit-data",
                str(selection),
                "--audit-output",
                str(args.output / label),
            ],
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"Execution audit failed for {label}; diagnostics are in {args.output}")
    summaries = {label: json.loads((args.output / label / "summary.json").read_text()) for label in checkpoints}
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)
