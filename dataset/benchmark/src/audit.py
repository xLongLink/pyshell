"""Grade next-script predictions by isolated execution, not text similarity."""

import json
import time
import httpx2
import asyncio
import argparse
from pathlib import Path
from sample.src.shell import RemoteShell
from sample.src.agents.script import STARTUP_LINES
from dataset.benchmark.src.cli import backend_lock
from dataset.benchmark.src.syntax import check_source
from dataset.benchmark.src.backend import ContainerBackend
from dataset.benchmark.src.recording import write_json
from dataset.benchmark.src.serialization import CODE, END_CODE, MARKERS, console_block

ROOT = Path(__file__).resolve().parents[1]


async def _initialize_shell(shell, record):
    bootstrap = await shell.stream("\n".join(STARTUP_LINES))
    setup = await shell.stream(record["setup"])
    if bootstrap.exception or setup.exception:
        raise RuntimeError(f"Fixture failed: {bootstrap.exception or setup.exception}")
    for previous in record["previous"]:
        await shell.stream(previous)


async def execute_candidate(backend, record, source):
    await backend.fresh()
    async with httpx2.AsyncClient(base_url=backend.url, timeout=60, trust_env=False) as client:
        shell = RemoteShell(client)
        async with asyncio.timeout(45):
            await _initialize_shell(shell, record)
            result = await shell.stream(source)
            if result.exception:
                return {"status": "execution_error", "error": result.exception, "done": result.done}
            if result.done is None:
                return {"status": "not_completed", "error": "Final next-script target did not call done()", "done": None}
            verification = await shell.stream(record["check"])
            if verification.exception:
                return {"status": "incorrect", "error": verification.exception, "done": result.done}
            return {"status": "passed", "error": None, "done": result.done}


async def rollout(backend, client, record, max_rounds):
    await backend.fresh()
    row = {"id": record["id"], "split": record["audit_split"], "family": record["family"], "steps": []}
    async with httpx2.AsyncClient(base_url=backend.url, timeout=60, trust_env=False) as shell_client:
        shell = RemoteShell(shell_client)
        async with asyncio.timeout(90):
            await _initialize_shell(shell, record)
            prompt = record["prompt"]
            for index in range(max_rounds):
                started = time.monotonic()
                response = await client.post(
                    "/completion",
                    json={
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0,
                        "n_predict": 512,
                        "stop": list(MARKERS),
                        "cache_prompt": True,
                    },
                )
                response.raise_for_status()
                completion = response.json()
                source = completion.get("content", "")
                error = check_source(source)
                ended = completion.get("stopping_word") == END_CODE and not completion.get("truncated")
                step = {
                    "source": source,
                    "syntax_valid": error is None,
                    "end_marker_seen": ended,
                    "generation_seconds": time.monotonic() - started,
                }
                row["steps"].append(step)
                if not ended or not source.strip():
                    row.update(status="boundary_error" if not ended else "empty_response", error=error, first_step_passed=False)
                    break
                result = await shell.stream(source)
                step["feedback"] = result.model_dump(exclude={"type"})
                if result.done is not None:
                    verification = await shell.stream(record["check"])
                    passed = verification.exception is None
                    row.update(
                        status="passed" if passed else "incorrect",
                        error=verification.exception,
                        done=result.done,
                        first_step_passed=passed and index == 0,
                    )
                    break
                prompt += source + "\n" + END_CODE + "\n" + console_block(step["feedback"]) + CODE + "\n"
            else:
                row.update(status="round_limit", error=f"No completion within {max_rounds} model calls", first_step_passed=False)
    row["syntax_valid"] = all(step["syntax_valid"] for step in row["steps"])
    return row


async def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in args.data.read_text().splitlines()]
    if not records:
        raise ValueError("Execution audit requires at least one record")
    backend = ContainerBackend(8020, args.output / "backend.log")
    summary = {
        "label": args.label,
        "scope": f"up to {args.max_rounds} code/console turns from reference prior context, verified by execution",
        "examples": 0,
        "splits": {},
        "hf_comparison": {
            "examples": 0,
            "hf_passed": 0,
            "gguf_passed": 0,
            "scope": "first-step completion correctness; HF gets one generation, not a rollout",
        },
    }
    async with httpx2.AsyncClient(
        base_url="http://127.0.0.1:8030", headers={"Authorization": "Bearer local-only"}, timeout=120, trust_env=False
    ) as client:
        try:
            for index, record in enumerate(records, 1):
                tokenized = await client.post("/tokenize", json={"content": record["prompt"], "add_special": False, "with_pieces": False})
                tokenized.raise_for_status()
                token_match = tokenized.json()["tokens"] == record["token_ids"]
                try:
                    row = await rollout(backend, client, record, args.max_rounds)
                except TimeoutError:
                    row = {
                        "id": record["id"],
                        "split": record["audit_split"],
                        "family": record["family"],
                        "steps": [],
                        "syntax_valid": False,
                        "status": "timeout",
                        "error": "Rollout exceeded 90 seconds",
                        "first_step_passed": False,
                    }
                row["tokenization_matches_hf"] = token_match
                if "hf_target_loss" in record:
                    row["hf_target_loss"] = record["hf_target_loss"]
                hf = record.get("hf_prediction")
                if hf is not None:
                    row["hf_prediction"] = hf
                    if check_source(hf["source"]) is None and hf["end_marker_seen"]:
                        try:
                            row["hf_execution"] = await execute_candidate(backend, record, hf["source"])
                        except TimeoutError:
                            row["hf_execution"] = {"status": "timeout", "error": "HF candidate execution exceeded 45 seconds"}
                with (args.output / "responses.jsonl").open("a") as file:
                    file.write(json.dumps(row) + "\n")
                summary["examples"] += 1
                if row["split"] not in summary["splits"]:
                    summary["splits"][row["split"]] = {
                        "examples": 0,
                        "passed": 0,
                        "syntax_valid": 0,
                        "generated_scripts": 0,
                        "valid_scripts": 0,
                        "tokenization_matches_hf": 0,
                        "hf_target_losses": [],
                        "by_family": {},
                    }
                    summary["splits"] = dict(sorted(summary["splits"].items()))
                split = summary["splits"][row["split"]]
                passed = row["status"] == "passed"
                split["examples"] += 1
                split["passed"] += passed
                split["syntax_valid"] += row["syntax_valid"]
                split["generated_scripts"] += len(row["steps"])
                split["valid_scripts"] += sum(step["syntax_valid"] for step in row["steps"])
                split["tokenization_matches_hf"] += token_match
                if "hf_target_loss" in row:
                    split["hf_target_losses"].append(row["hf_target_loss"])
                if row["family"] not in split["by_family"]:
                    split["by_family"][row["family"]] = {"examples": 0, "passed": 0}
                    split["by_family"] = dict(sorted(split["by_family"].items()))
                family = split["by_family"][row["family"]]
                family["examples"] += 1
                family["passed"] += passed
                if "hf_prediction" in row:
                    comparison = summary["hf_comparison"]
                    comparison["examples"] += 1
                    comparison["hf_passed"] += row.get("hf_execution", {}).get("status") == "passed"
                    comparison["gguf_passed"] += row["first_step_passed"]
                if index % 10 == 0:
                    passed_total = sum(split["passed"] for split in summary["splits"].values())
                    print(f"Execution audit {index}/{len(records)}: {passed_total} passed", flush=True)
                write_json(args.output / "summary.json", summary)
        finally:
            await backend.close()
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-rounds", type=int, default=4)
    args = parser.parse_args(argv)
    with backend_lock():
        return asyncio.run(evaluate(args))
