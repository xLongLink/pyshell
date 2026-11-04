"""Execute cases through the production worker and grade their Python state."""

import httpx2
import asyncio
from time import monotonic
from typing import Any
from pathlib import Path
from contextlib import redirect_stdout
from dataclasses import dataclass
from pydantic_ai.models import Model
from pydantic_ai.exceptions import ModelAPIError
from sample.src.agents.script import Agent
from dataset.benchmark.src.cases import Case
from dataset.benchmark.src.recording import Trace, TokenBudget, Measurements, MeasuredModel, MeasuredShell, BudgetExceeded, write_json


@dataclass(frozen=True)
class Limits:
    """Per-case bounds shared by all follow-up instructions."""

    seconds: float = 120
    calls: int = 12
    tokens: int = 30_000


async def run_trial(
    case: Case, *, repeat: int, model: Model, shell_url: str, directory: Path, limits: Limits, budget: TokenBudget
) -> dict[str, Any]:
    """Run and persist one isolated trial; callers own container recreation."""
    directory.mkdir(parents=True)
    trace = Trace(directory / "trace.jsonl")
    measurements = Measurements()
    trial: dict[str, Any] = {
        "case": case.id,
        "category": case.category,
        "repeat": repeat,
        "status": "infrastructure_error",
        "error": None,
        "completed_steps": 0,
        "steps": len(case.steps),
        "task_seconds": 0.0,
        "initialization_seconds": None,
        "verification_seconds": 0.0,
        "done": [],
        "uncertain": False,
    }
    trace.write(
        "case", id=case.id, setup=case.setup, steps=[{"instruction": step.instruction, "verify": step.verify} for step in case.steps]
    )
    started = monotonic()
    task_started: float | None = None
    stage = "setup"
    interrupted = False
    async with httpx2.AsyncClient(base_url=shell_url, timeout=httpx2.Timeout(180, connect=10), trust_env=False) as client:
        shell = MeasuredShell(client, trace=trace, measurements=measurements)
        measured_model = MeasuredModel(
            model, trace=trace, measurements=measurements, budget=budget, max_calls=limits.calls, max_tokens=limits.tokens
        )
        agent = Agent(measured_model, shell=shell)

        def phase_changed(phase: str) -> None:
            nonlocal task_started
            if phase == "generating_code":
                if task_started is None:
                    task_started = monotonic()
                shell.phase = "worker"

        try:
            with (directory / "worker.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
                async with asyncio.timeout(limits.seconds):
                    await agent.reset()
                    result = await shell.stream(case.setup)
                    if result.exception or result.more_input or result.done is not None:
                        raise RuntimeError(f"Case setup failed: {result.model_dump_json()}")
                    shell.phase = "initialization"
                    for step in case.steps:
                        stage = "worker"
                        turn_started = monotonic()
                        task_started = None
                        try:
                            completion = await agent.arun(step.instruction, phase_changed)
                        finally:
                            finished = monotonic()
                            if trial["initialization_seconds"] is None:
                                trial["initialization_seconds"] = (task_started or finished) - turn_started
                            if task_started is not None:
                                trial["task_seconds"] += finished - task_started
                        trial["done"].append(completion)
                        stage = "verification"
                        shell.phase = "verification"
                        verification_started = monotonic()
                        result = await shell.stream(step.verify)
                        trial["verification_seconds"] += monotonic() - verification_started
                        if result.exception or result.more_input or result.done is not None:
                            trial["status"] = "incorrect"
                            trial["error"] = result.exception or "Verification did not complete cleanly"
                            break
                        trial["completed_steps"] += 1
                    else:
                        trial["status"] = "passed"
        except TimeoutError:
            trial["status"] = "timeout"
            trial["error"] = f"Trial exceeded {limits.seconds} seconds during {stage}"
        except BudgetExceeded as error:
            trial["status"] = "budget_exceeded"
            trial["error"] = str(error)
        except ModelAPIError as error:
            trial["status"] = "provider_error"
            trial["error"] = str(error)
        except asyncio.CancelledError:
            trial["status"] = "interrupted"
            trial["error"] = f"Cancelled during {stage}"
            interrupted = True
        except Exception as error:  # noqa: BLE001
            trial["status"] = "infrastructure_error" if shell.uncertain or stage != "worker" or not agent.initialized else "worker_error"
            trial["error"] = f"{type(error).__name__}: {error}"
        finally:
            trial["uncertain"] = shell.uncertain
    trial.update(
        wall_seconds=monotonic() - started,
        input_tokens=measurements.input_tokens,
        output_tokens=measurements.output_tokens,
        calls=len(measurements.calls),
        call_timings=measurements.calls,
        execution_errors=sum(execution["phase"] == "worker" and execution["error"] for execution in measurements.executions),
        worker_execution_seconds=sum(execution["seconds"] for execution in measurements.executions if execution["phase"] == "worker"),
    )
    trace.write("trial_result", **trial)
    write_json(directory / "trial.json", trial)
    if interrupted:
        raise asyncio.CancelledError
    return trial
