"""CLI: uv run --project model/benchmark python -m model.benchmark.main --help."""

import os
import sys
import json
import uuid
import asyncio
import hashlib
import argparse
import subprocess
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import Any
from pathlib import Path
from datetime import UTC, datetime
from contextlib import AsyncExitStack
from dataclasses import asdict
from importlib.metadata import version
from pyshell.serialization import FORMAT
from model.benchmark.src.cases import CASES
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from model.benchmark.src.report import summarize
from model.benchmark.src.runner import Limits, run_trial
from model.benchmark.src.backend import ContainerBackend, backend_lock
from pydantic_ai.providers.openai import OpenAIProvider
from model.benchmark.src.raw_model import RawCompletionModel
from model.benchmark.src.recording import TokenBudget, write_json

ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    limits = Limits()
    cli = argparse.ArgumentParser(
        description="Evaluate the Python worker, without voice or UI grading. Run from the repository root.",
        epilog="Traces contain prompts, Python, and execution output. Use synthetic data. No API calls are retried automatically.",
    )
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the predefined cases without Docker or API credentials")
    report = commands.add_parser("report", help="Summarize saved trial records without rerunning a model")
    report.add_argument("directory", type=Path)
    run = commands.add_parser("run", help="Run real worker tasks and verify the Python results")
    run.add_argument("--model", required=True, help="Provider's model ID")
    run.add_argument("--label", help="Human-readable model/configuration label")
    run.add_argument("--api", choices=("chat", "responses", "raw"), default="chat")
    run.add_argument("--base-url", help="OpenAI-compatible provider base URL; defaults to OpenAI")
    run.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable holding the provider key")
    run.add_argument(
        "--env-file", type=Path, help="Dotenv file (default: model/benchmark/.env); existing environment variables take precedence"
    )
    run.add_argument("--settings", default="{}", help='Pydantic AI model settings as JSON, e.g. {"openai_reasoning_effort":"low"}')
    run.add_argument("--max-output-tokens", type=positive_int, default=2048)
    run.add_argument("--case", action="append", choices=[case.id for case in CASES], help="Select a case; repeat to select several")
    run.add_argument("--repeats", type=positive_int, default=1)
    run.add_argument(
        "--max-seconds", type=positive_int, default=limits.seconds, help="Trial deadline, including initialization and verification"
    )
    run.add_argument("--max-calls", type=positive_int, default=limits.calls, help="Shared across all turns of one case")
    run.add_argument("--max-trial-tokens", type=positive_int, default=limits.tokens)
    run.add_argument(
        "--max-run-tokens", type=positive_int, default=300_000, help="Observed tokens; may overshoot by one in-flight response"
    )
    run.add_argument("--output", type=Path, default=ROOT / "results")
    run.add_argument("--port", type=positive_int, default=8020, help="Dedicated managed-container port")
    run.add_argument("--no-build", action="store_true", help="Reuse an already built benchmark image")
    run.add_argument("--shell-url", help="Use an existing dedicated backend instead of managing Docker; reset-only isolation")
    run.add_argument("--allow-reset", action="store_true", help="Required with --shell-url: confirms its workspace may be erased")
    return cli


def print_summary(summary: dict[str, Any]) -> None:
    timings = summary["successful_task_seconds"]
    median = f"{timings['median']:.2f}s" if timings["median"] is not None else "n/a"
    p95 = f"{timings['p95']:.2f}s" if timings["p95"] is not None else "n/a"
    print(f"Passed {summary['passed']}/{summary['trials']} | successful task median {median}, p95 {p95}")
    for name, label in (("first_text_seconds", "First text"), ("complete_code_seconds", "Complete code")):
        timing = summary[name]
        if timing["median"] is not None:
            print(f"{label}: median {timing['median']:.3f}s, p95 {timing['p95']:.3f}s")
    if summary["mean_calls"] is not None:
        print(f"Mean model calls per trial: {summary['mean_calls']:.2f}")
    print(f"Statuses: {summary['statuses']} | tokens: {summary['input_tokens']} input, {summary['output_tokens']} output")
    if summary["unmeasured_calls"]:
        print(f"Warning: {summary['unmeasured_calls']} calls have missing usage; token budgets cannot bound their cost.")


async def run(args: argparse.Namespace) -> int:
    env_file = args.env_file or ROOT / ".env"
    if args.env_file and not env_file.is_file():
        raise ValueError(f"Environment file does not exist: {env_file}")
    if env_file.is_file():
        load_dotenv(env_file, override=False)
    key = os.environ.get(args.api_key_env)
    if not key:
        raise ValueError(f"Set {args.api_key_env} before running the benchmark")
    if args.shell_url and not args.allow_reset:
        raise ValueError("--shell-url requires --allow-reset; never point it at an interactive workspace")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    if "@" in base_url or "?" in base_url or "#" in base_url:
        raise ValueError("--base-url must not contain credentials, query parameters, or fragments")
    settings = json.loads(args.settings)
    if not isinstance(settings, dict):
        raise TypeError("--settings must be a JSON object")
    settings["max_tokens"] = args.max_output_tokens
    settings["timeout"] = args.max_seconds
    selected = [case for case in CASES if not args.case or case.id in args.case]
    limits = Limits(seconds=args.max_seconds, calls=args.max_calls, tokens=args.max_trial_tokens)
    budget = TokenBudget(args.max_run_tokens)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = args.output.resolve() / run_id
    directory.mkdir(parents=True)
    revision = await asyncio.to_thread(subprocess.run, ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    from sample.src.agents.script import INSTRUCTIONS, STARTUP_LINES

    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "label": args.label or args.model,
        "model": args.model,
        "api": args.api,
        "base_url": base_url,
        "settings": settings,
        "limits": asdict(limits),
        "max_run_tokens": budget.limit,
        "cases": [case.id for case in selected],
        "repeats": args.repeats,
        "planned_trials": len(selected) * args.repeats,
        "isolation": "reset_only" if args.shell_url else "fresh_container",
        "revision": revision.stdout.strip() or None,
        "suite_sha256": hashlib.sha256((ROOT / "src/cases.py").read_bytes()).hexdigest(),
        "worker_sha256": hashlib.sha256((ROOT.parents[1] / "sample/src/agents/script.py").read_bytes()).hexdigest(),
        "instructions": None if args.api == "raw" else INSTRUCTIONS,
        "serialization": FORMAT if args.api == "raw" else "chat",
        "startup_lines": STARTUP_LINES,
        "pydantic_ai_version": version("pydantic-ai-slim"),
        "stopped_reason": None,
    }
    write_json(directory / "run.json", metadata)
    print(f"Results: {directory}")
    print(f"Running {len(selected)} cases × {args.repeats} repeats; grading Python state, not UI.")
    backend = ContainerBackend(args.port, directory / "backend.log") if not args.shell_url else None
    trials: list[dict[str, Any]] = []

    def save() -> None:
        write_json(directory / "run.json", metadata)
        write_json(directory / "summary.json", summarize(trials))

    try:
        if backend and not args.no_build:
            print("Building dedicated execution image…")
            await backend.build()
        async with AsyncExitStack() as resources:
            if args.api == "raw":
                model = await resources.enter_async_context(
                    RawCompletionModel(args.model, base_url=base_url, api_key=key, settings=settings)
                )
            else:
                client = await resources.enter_async_context(
                    AsyncOpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=args.max_seconds)
                )
                provider = OpenAIProvider(openai_client=client)
                model_type = OpenAIResponsesModel if args.api == "responses" else OpenAIChatModel
                model = await resources.enter_async_context(model_type(args.model, provider=provider, settings=settings))
            for repeat in range(1, args.repeats + 1):
                for case in selected:
                    if budget.used >= budget.limit:
                        metadata["stopped_reason"] = "run_token_budget"
                        break
                    if backend:
                        await backend.fresh()
                    trial_directory = directory / f"{case.id}-{repeat:03d}"
                    trial = await run_trial(
                        case,
                        repeat=repeat,
                        model=model,
                        shell_url=backend.url if backend else args.shell_url,
                        directory=trial_directory,
                        limits=limits,
                        budget=budget,
                    )
                    trials.append(trial)
                    save()
                    print(f"{case.id} [{repeat}/{args.repeats}]: {trial['status']} ({trial['task_seconds']:.2f}s, {trial['calls']} calls)")
                    if trial["status"] in {"provider_error", "infrastructure_error"} or (trial["uncertain"] and not backend):
                        metadata["stopped_reason"] = trial["status"]
                        break
                if metadata["stopped_reason"]:
                    break
    except BaseException as error:
        metadata["stopped_reason"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
        if backend:
            await backend.close()
    summary = summarize(trials)
    print_summary(summary)
    return 0 if len(trials) == metadata["planned_trials"] and all(trial["status"] == "passed" for trial in trials) else 1


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "list":
        for case in CASES:
            print(f"{case.id:24} {case.category:12} {len(case.steps)} turn(s)")
        return 0
    if args.command == "report":
        trials = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.directory.glob("*/trial.json"))]
        if not trials:
            print("No trial.json records found", file=sys.stderr)
            return 2
        summary = summarize(trials)
        write_json(args.directory / "summary.json", summary)
        print_summary(summary)
        return 0
    try:
        with backend_lock():
            return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Benchmark interrupted; completed traces remain on disk.", file=sys.stderr)
        return 130
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2
