"""Measure unmodified response compilability without executing generated code."""

import ast
import json
import time
import uuid
import argparse
import urllib.request
from pathlib import Path
from collections import Counter
from pyshell.serialization import MARKERS, END_CODE
from model.benchmark.src.corpus import default_training_data

ROOT = Path(__file__).resolve().parents[1]


def check_source(source):
    if not source or not source.strip():
        return "empty_response"
    try:
        tree = ast.parse(source, filename="<candidate>", mode="exec")
        compile(tree, "<candidate>", "exec")
    except (SyntaxError, ValueError, TypeError) as error:
        return f"{type(error).__name__}: {error}"
    return None if tree.body else "no_python_statements"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--serialization", choices=("chat", "raw"), default="raw")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or ROOT / "logs" / f"syntax-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    data_path = args.data or default_training_data(args.serialization) / "validation.jsonl"
    records = [json.loads(line) for line in data_path.read_text().splitlines()]
    compiles = 0
    passed = 0
    finish_reasons = Counter()
    per_family = {}
    print(f"Syntax-only evaluation: {len(records)} held-out prompts, temperature={args.temperature}; results: {output}", flush=True)
    for index, record in enumerate(records, 1):
        payload = {
            "model": args.model,
            "messages": record["messages"][:-1] if args.serialization == "chat" else [],
            "temperature": args.temperature,
            "top_p": 0.8,
            "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        endpoint = "http://127.0.0.1:8030/v1/chat/completions"
        if args.serialization == "raw":
            endpoint = "http://127.0.0.1:8030/completion"
            payload = {
                "prompt": record["prompt"],
                "temperature": args.temperature,
                "top_p": 0.8,
                "n_predict": 2048,
                "stop": list(MARKERS),
                "stream": False,
                "cache_prompt": True,
            }
        request = urllib.request.Request(
            endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer local-only"}
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=120) as response:
            completion = json.load(response)
        if args.serialization == "raw":
            source = completion.get("content") or ""
            finish_reason = (
                "stop" if completion.get("stopping_word") == END_CODE and not completion.get("truncated") else "missing_end_marker"
            )
        else:
            choice = completion["choices"][0]
            source = choice["message"].get("content") or ""
            finish_reason = choice["finish_reason"]
        error = check_source(source)
        item = {
            "id": record["id"],
            "family": record["family"],
            "source": source,
            "compile_error": error,
            "compiles": error is None,
            "finish_reason": finish_reason,
            "passed": error is None and finish_reason == "stop",
            "seconds": time.monotonic() - started,
            "usage": completion.get("usage"),
        }

        # Keep aggregate outcomes; full responses are persisted below.
        compiles += item["compiles"]
        passed += item["passed"]
        finish_reasons[finish_reason] += 1
        family = per_family.setdefault(item["family"], {"total": 0, "passed": 0})
        family["total"] += 1
        family["passed"] += item["passed"]
        with (output / "responses.jsonl").open("a") as file:
            file.write(json.dumps(item) + "\n")
        if index % 10 == 0 or not item["passed"]:
            print(f"{index}/{len(records)}: {passed} valid completed responses; latest: {error or finish_reason}", flush=True)
    summary = {
        "model": args.model,
        "temperature": args.temperature,
        "serialization": args.serialization,
        "prompts": len(records),
        "compiles": compiles,
        "passed": passed,
        "raw_python_rate": compiles / len(records),
        "completed_python_rate": passed / len(records),
        "finish_reasons": dict(finish_reasons),
        "per_family": dict(sorted(per_family.items())),
        "scope": "raw syntax only; no stripping, repairs, execution, or task correctness grading",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if passed == len(records) else 1
