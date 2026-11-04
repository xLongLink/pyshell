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
from dataset.benchmark.src.serialization import CODE, TASK, CONSOLE, END_CODE

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
                        "stop": [END_CODE, TASK, CONSOLE, CODE],
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
                prompt += source + "\n" + END_CODE + "\n" + CONSOLE + "\n"
                prompt += json.dumps(step["feedback"], ensure_ascii=False).replace("<", "\\u003c") + "\n" + CODE + "\n"
            else:
                row.update(status="round_limit", error=f"No completion within {max_rounds} model calls", first_step_passed=False)
    row["syntax_valid"] = all(step["syntax_valid"] for step in row["steps"])
    return row


async def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in args.data.read_text().splitlines()]
    backend = ContainerBackend(8020, args.output / "backend.log")
    results = []
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
                results.append(row)
                with (args.output / "responses.jsonl").open("a") as file:
                    file.write(json.dumps(row) + "\n")
                if index % 10 == 0:
                    print(f"Execution audit {index}/{len(records)}: {sum(r['status'] == 'passed' for r in results)} passed", flush=True)
                summary = {
                    "label": args.label,
                    "scope": f"up to {args.max_rounds} code/console turns from reference prior context, verified by execution",
                    "examples": len(results),
                    "splits": {},
                }
                for split in sorted({r["split"] for r in results}):
                    group = [r for r in results if r["split"] == split]
                    summary["splits"][split] = {
                        "examples": len(group),
                        "passed": sum(r["status"] == "passed" for r in group),
                        "syntax_valid": sum(r["syntax_valid"] for r in group),
                        "generated_scripts": sum(len(r["steps"]) for r in group),
                        "valid_scripts": sum(step["syntax_valid"] for r in group for step in r["steps"]),
                        "tokenization_matches_hf": sum(r["tokenization_matches_hf"] for r in group),
                        "hf_target_losses": [r["hf_target_loss"] for r in group if "hf_target_loss" in r],
                        "by_family": {
                            family: {
                                "examples": sum(r["family"] == family for r in group),
                                "passed": sum(r["family"] == family and r["status"] == "passed" for r in group),
                            }
                            for family in sorted({r["family"] for r in group})
                        },
                    }
                paired = [r for r in results if "hf_prediction" in r]
                summary["hf_comparison"] = {
                    "examples": len(paired),
                    "hf_passed": sum(r.get("hf_execution", {}).get("status") == "passed" for r in paired),
                    "gguf_passed": sum(r["first_step_passed"] for r in paired),
                    "scope": "first-step completion correctness; HF gets one generation, not a rollout",
                }
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
