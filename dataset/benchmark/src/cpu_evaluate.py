"""Measure fresh-workspace execution through a network-isolated Python container."""

import json
import argparse
import urllib.request
from pathlib import Path
from contextlib import closing, contextmanager
from model.train.src.resources import exclusive_cpu
from dataset.benchmark.src.corpus import sha256, load_tasks
from dataset.benchmark.src.execution import Worker
from dataset.benchmark.src.proof_trial import trial
from dataset.benchmark.src.proof_policy import final_passed
from dataset.benchmark.src.proof_report import execution_metrics
from dataset.benchmark.src.serialization import CODE, TASK, CONSOLE, END_CODE
from dataset.benchmark.src.inference_server import inference_server

ROOT = Path(__file__).resolve().parents[1]


def completion(prompt):
    payload = {
        "prompt": prompt,
        "temperature": 0,
        "seed": 713,
        "n_predict": 384,
        "stop": [END_CODE, TASK, CONSOLE, CODE],
        "stream": False,
        "cache_prompt": True,
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8032/completion", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.load(response)
    return result["content"], result.get("stopping_word") == END_CODE


@contextmanager
def generator(args):
    if args.backend == "hf":
        from model.train.src.inference import HFCompletion

        yield HFCompletion(args.adapter, args.threads)
        return
    command = [
        str(ROOT / "local/llama-b12038/llama-b12038/llama-server"),
        "--model",
        str(args.gguf_base.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        "8032",
        "--ctx-size",
        "4096",
        "--parallel",
        "1",
        "--threads",
        "8",
        "--threads-batch",
        "8",
        "--n-gpu-layers",
        "0",
    ]
    if args.adapter:
        command += ["--lora", str(args.adapter.resolve())]
    with inference_server(command, args.output / "server.log", port=8032):
        yield completion


def evaluate(args):
    manifest, digest, records = load_tasks(args.data, args.split, final=args.final, limit=args.limit)
    adapter_file = None
    if args.adapter:
        adapter_file = args.adapter / "adapter_model.safetensors" if args.adapter.is_dir() else args.adapter
    adapter_digest = sha256(adapter_file) if adapter_file else None
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    with exclusive_cpu(), generator(args) as generate, closing(Worker()) as worker:
        for index, record in enumerate(records, 1):
            row = trial(worker, record, generate)
            rows.append(row)
            with (args.output / "responses.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            metrics = execution_metrics(rows)
            summary = {
                "label": args.label,
                "split": args.split,
                "complete": len(rows) == len(records),
                **metrics,
                "task_file_sha256": digest,
                "adapter": str(args.adapter) if args.adapter else None,
                "adapter_sha256": adapter_digest,
                "backend": args.backend,
                "gguf_base": str(args.gguf_base) if args.backend == "gguf" else None,
                "scope": manifest["scope"],
                "isolation": "network-disabled container; fresh PythonShell per task; no reference prior context",
                "acceptance_met": args.split == "test" and final_passed(len(rows), metrics["passed"]),
            }
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(
                f"{args.label} {args.split} {index}/{len(records)}: {metrics['passed']} passed; {record['family']}={row['status']}",
                flush=True,
            )
    if adapter_file and sha256(adapter_file) != adapter_digest:
        raise RuntimeError("Adapter changed during evaluation")
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/cpu-proof-v1")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backend", choices=("gguf", "hf"), default="gguf")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gguf-base", type=Path, default=ROOT / "local/Qwen3.5-0.8B-Q4_K_M.gguf")
    evaluate(parser.parse_args(argv))
