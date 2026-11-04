"""Exercise the real worker with a deterministic model and in-memory transport."""

import io
import json
import httpx2
import asyncio
import unittest
from typing import Any
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stderr, redirect_stdout, asynccontextmanager
from unittest.mock import patch
from sample.src.agents.script import STARTUP_LINES
from pydantic_ai.models.wrapper import WrapperModel
from dataset.benchmark.src.cases import Case, Step
from pydantic_ai.models.function import FunctionModel
from dataset.benchmark.src.report import summarize, distribution
from dataset.benchmark.src.runner import Limits, run_trial
from dataset.benchmark.src.recording import Trace, Measurements, MeasuredShell, TokenBudget


class StoppedModel(WrapperModel):
    """FunctionModel has no provider finish reason; supply the worker's contract."""

    @asynccontextmanager
    async def request_stream(self, *args: Any, **kwargs: Any):
        async with super().request_stream(*args, **kwargs) as stream:
            stream.finish_reason = "stop"
            yield stream


class ExecutionTransport:
    """Minimal execution protocol for harness tests, not an agent simulator."""

    def __init__(self) -> None:
        self.namespace: dict[str, Any] = {}
        self.submitted: list[str] = []
        self.finished = None
        self.disconnect = False
        self.reset()

    def reset(self) -> None:
        self.namespace = {"done": self.done}

    def done(self, message: str) -> None:
        self.finished = message

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/reset":
            self.reset()
            return httpx2.Response(200, json={"status": "reset"})
        code = json.loads(request.content)["code"]
        self.submitted.append(code)
        if self.disconnect:
            raise httpx2.ReadError("Disconnected", request=request)
        stdout, stderr = io.StringIO(), io.StringIO()
        error = None
        self.finished = None
        if code != "\n".join(STARTUP_LINES):
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exec(code, self.namespace)
            except Exception as exception:  # noqa: BLE001
                error = f"{type(exception).__name__}: {exception}"
                self.finished = None
        events = [
            {"type": "result", "stdout": stdout.getvalue(), "stderr": stderr.getvalue(), "exception": error, "done": self.finished},
            {"type": "execution_finished"},
        ]
        return httpx2.Response(200, text="\n".join(json.dumps(event) for event in events))


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    async def trial(
        self,
        case: Case,
        scripts: list[str],
        *,
        calls: int = 12,
        tokens: int = 30_000,
        seconds: float = 5,
        delay: float = 0,
        transport: ExecutionTransport | None = None,
        budget: TokenBudget | None = None,
    ):
        self.transport = transport or ExecutionTransport()
        self.requests = []
        responses = iter(scripts)

        async def stream(messages, _info):
            self.requests.append(messages)
            if delay:
                await asyncio.sleep(delay)
            code = next(responses)
            yield code[:5]
            yield code[5:]

        model = StoppedModel(FunctionModel(stream_function=stream))
        original_client = httpx2.AsyncClient

        def client(**kwargs):
            return original_client(**kwargs, transport=httpx2.MockTransport(self.transport))

        with TemporaryDirectory(dir="/tmp/opencode") as temporary:
            directory = Path(temporary) / "trial"
            with patch("dataset.benchmark.src.runner.httpx2.AsyncClient", side_effect=client):
                result = await run_trial(
                    case,
                    repeat=1,
                    model=model,
                    shell_url="http://testserver",
                    directory=directory,
                    limits=Limits(seconds=seconds, calls=calls, tokens=tokens),
                    budget=budget if budget is not None else TokenBudget(1_000_000),
                )
            self.saved = json.loads((directory / "trial.json").read_text())
            self.trace = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
        return result

    async def test_success_records_timing_usage_and_hidden_verification(self):
        case = Case("sum", "computation", "values = [1, 2, 3]", (Step("Sum values into result, then done().", "assert result == 6"),))
        result = await self.trial(case, ["result = sum(values)\ndone('computed')"])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(self.saved, result)
        self.assertEqual(result["calls"], 1)
        self.assertGreater(result["output_tokens"], 0)
        self.assertIsNotNone(result["call_timings"][0]["first_text_seconds"])
        self.assertGreaterEqual(result["call_timings"][0]["total_seconds"], result["call_timings"][0]["first_text_seconds"])
        request_events = [event for event in self.trace if event["event"] == "model_request"]
        self.assertNotIn("assert result == 6", json.dumps(request_events))
        self.assertEqual(
            [event["phase"] for event in self.trace if event["event"] == "execution_request"],
            ["setup", "initialization", "worker", "verification"],
        )

    async def test_false_success_fails(self):
        case = Case("wrong", "computation", "", (Step("Compute result", "assert result == 6"),))
        result = await self.trial(case, ["result = 99\ndone('success')"])
        self.assertEqual(result["status"], "incorrect")
        self.assertEqual(result["completed_steps"], 0)

    async def test_execution_error_can_recover(self):
        case = Case("recover", "recovery", "", (Step("Compute result", "assert result == 6"),))
        result = await self.trial(case, ["raise ValueError('retry')", "result = 6\ndone('recovered')"])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["execution_errors"], 1)
        self.assertEqual(result["calls"], 2)
        executions = [event for event in self.trace if event["event"] == "execution_result" and event["phase"] == "worker"]
        self.assertEqual(
            executions[0]["result"],
            {"type": "result", "stdout": "", "stderr": "", "exception": "ValueError: retry", "done": None, "more_input": False},
        )
        self.assertEqual(result["worker_execution_seconds"], sum(event["seconds"] for event in executions))

    async def test_empty_exception_still_counts_as_an_execution_error(self):
        class EmptyExceptionTransport(ExecutionTransport):
            def __call__(self, request):
                response = super().__call__(request)
                if request.url.path == "/execute/stream":
                    events = [json.loads(line) for line in response.text.splitlines()]
                    if events[0]["exception"] is not None:
                        events[0]["exception"] = ""
                    return httpx2.Response(200, text="\n".join(json.dumps(event) for event in events))
                return response

        case = Case("empty_exception", "recovery", "", (Step("Compute result", "assert result == 6"),))
        result = await self.trial(case, ["raise ValueError()", "result = 6\ndone('recovered')"], transport=EmptyExceptionTransport())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["execution_errors"], 1)
        executions = [event for event in self.trace if event["event"] == "execution_result" and event["phase"] == "worker"]
        self.assertEqual(executions[0]["result"]["exception"], "")

    async def test_followups_share_state_but_not_verifier_feedback(self):
        case = Case(
            "state",
            "state",
            "values = []",
            (
                Step("Append 1", "assert values == [1]\noriginal = values"),
                Step("Append 2", "assert values == [1, 2]\nassert values is original"),
            ),
        )
        result = await self.trial(case, ["values.append(1)\ndone('one')", "values.append(2)\ndone('two')"])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["completed_steps"], 2)
        request_events = [event for event in self.trace if event["event"] == "model_request"]
        self.assertNotIn("original = values", json.dumps(request_events))

    async def test_call_limit_is_shared_across_followups(self):
        case = Case("limit", "state", "", (Step("First", "pass"), Step("Second", "pass")))
        result = await self.trial(case, ["done('one')", "done('two')"], calls=1)
        self.assertEqual(result["status"], "budget_exceeded")
        self.assertEqual(result["calls"], 1)
        self.assertEqual(result["completed_steps"], 1)

    async def test_token_overshoot_does_not_execute_generated_code(self):
        case = Case("tokens", "computation", "", (Step("Compute", "pass"),))
        result = await self.trial(case, ["result = 6\ndone('six')"], tokens=1)
        self.assertEqual(result["status"], "budget_exceeded")
        self.assertNotIn("result", self.transport.namespace)

    async def test_generation_timeout_is_not_execution_uncertainty(self):
        case = Case("timeout", "computation", "", (Step("Compute", "pass"),))
        result = await self.trial(case, ["done('late')"], seconds=0.02, delay=0.1)
        self.assertEqual(result["status"], "timeout")
        self.assertFalse(result["uncertain"])

    async def test_run_budget_is_shared_between_trials(self):
        case = Case("run_budget", "computation", "", (Step("Compute", "pass"),))
        budget = TokenBudget(1)
        first = await self.trial(case, ["done('one')"], budget=budget)
        self.assertEqual(first["status"], "budget_exceeded")
        self.assertGreater(budget.used, budget.limit)
        second = await self.trial(case, [], budget=budget)
        self.assertEqual(second["status"], "budget_exceeded")
        self.assertEqual(second["calls"], 0)

    async def test_transport_failure_requires_backend_recreation(self):
        transport = ExecutionTransport()
        transport.disconnect = True
        case = Case("connection", "computation", "", (Step("Compute", "pass"),))
        result = await self.trial(case, [], transport=transport)
        self.assertEqual(result["status"], "infrastructure_error")
        self.assertTrue(result["uncertain"])
        self.assertEqual(result["execution_errors"], 0)
        execution = next(event for event in self.trace if event["event"] == "execution_result")
        self.assertEqual(set(execution), {"event", "phase", "seconds", "result"})
        self.assertIsNone(execution["result"])


class MeasuredShellTests(unittest.IsolatedAsyncioTestCase):
    async def test_compact_measurements_preserve_full_result_traces(self):
        for exception in (None, "", "ValueError: failed"):
            with self.subTest(exception=exception), TemporaryDirectory(dir="/tmp/opencode") as temporary:
                payload = {
                    "type": "result",
                    "stdout": "captured output\n" * 1000,
                    "stderr": "captured error\n",
                    "exception": exception,
                    "done": "finished" if exception is None else None,
                    "more_input": False,
                }

                def respond(request):
                    return httpx2.Response(200, text=json.dumps(payload) + '\n{"type": "execution_finished"}')

                measurements = Measurements()
                trace = Trace(Path(temporary) / "trace.jsonl")
                async with httpx2.AsyncClient(base_url="http://testserver", transport=httpx2.MockTransport(respond)) as client:
                    shell = MeasuredShell(client, trace=trace, measurements=measurements)
                    shell.phase = "worker"
                    result = await shell.stream("print('output')")
                self.assertEqual(result.model_dump(), payload)
                self.assertEqual(len(measurements.executions), 1)
                execution = measurements.executions[0]
                self.assertEqual(execution, {"phase": "worker", "seconds": execution["seconds"], "error": exception is not None})
                self.assertGreater(execution["seconds"], 0)
                events = [json.loads(line) for line in trace.path.read_text().splitlines()]
                self.assertEqual(
                    events[-1], {"event": "execution_result", "phase": "worker", "seconds": execution["seconds"], "result": payload}
                )

    async def test_transport_failure_and_cancellation_are_timed_without_execution_errors(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel), TemporaryDirectory(dir="/tmp/opencode") as temporary:

                async def respond(request):
                    if cancel:
                        await asyncio.Future()
                    raise httpx2.ReadError("Disconnected", request=request)

                measurements = Measurements()
                trace = Trace(Path(temporary) / "trace.jsonl")
                async with httpx2.AsyncClient(base_url="http://testserver", transport=httpx2.MockTransport(respond)) as client:
                    shell = MeasuredShell(client, trace=trace, measurements=measurements)
                    shell.phase = "worker"
                    with self.assertRaises(TimeoutError if cancel else RuntimeError):
                        async with asyncio.timeout(0.01):
                            await shell.stream("pass")
                self.assertTrue(shell.uncertain)
                self.assertEqual(len(measurements.executions), 1)
                execution = measurements.executions[0]
                self.assertEqual(execution, {"phase": "worker", "seconds": execution["seconds"], "error": False})
                self.assertGreater(execution["seconds"], 0)
                if cancel:
                    self.assertGreaterEqual(execution["seconds"], 0.01)
                events = [json.loads(line) for line in trace.path.read_text().splitlines()]
                self.assertEqual(
                    events[-1], {"event": "execution_result", "phase": "worker", "seconds": execution["seconds"], "result": None}
                )


class MeasurementTests(unittest.TestCase):
    def test_totals_follow_recorded_usage_including_failed_calls(self):
        measurements = Measurements()
        self.assertEqual((measurements.input_tokens, measurements.output_tokens), (0, 0))
        pending = {"usage": None, "error": None}
        measurements.calls.extend(
            [
                {"usage": {"input_tokens": 10, "output_tokens": 3}, "error": None},
                {"usage": {"input_tokens": 7, "output_tokens": 2}, "error": "ReadError"},
                pending,
            ]
        )
        self.assertEqual((measurements.input_tokens, measurements.output_tokens), (17, 5))
        pending["usage"] = {"input_tokens": 4, "output_tokens": 1}
        self.assertEqual((measurements.input_tokens, measurements.output_tokens), (21, 6))


class ReportTests(unittest.TestCase):
    def test_nearest_rank_p95(self):
        self.assertEqual(distribution([1, 2, 3, 4, 5]), {"median": 3, "p95": 5})
        self.assertEqual(distribution([]), {"median": None, "p95": None})

    def test_failed_fast_trials_do_not_improve_success_latency(self):
        common = {"case": "a", "calls": 1, "call_timings": [], "input_tokens": 1, "output_tokens": 1}
        result = summarize([{**common, "status": "passed", "task_seconds": 10}, {**common, "status": "incorrect", "task_seconds": 0.1}])
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["successful_task_seconds"]["median"], 10)
        self.assertEqual(result["per_case"]["a"]["statuses"], {"passed": 1, "incorrect": 1})
