"""Streaming measurements and durable, credential-free execution traces."""

import json
from time import monotonic
from typing import Any, cast
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import field, asdict, dataclass
from pydantic_ai import RunContext
from collections.abc import AsyncIterator, AsyncGenerator
from sample.src.shell import Result, RemoteShell
from pydantic_ai.models import StreamedResponse, ModelRequestParameters
from pydantic_ai.messages import (
    TextPart,
    ModelMessage,
    TextPartDelta,
    PartDeltaEvent,
    PartStartEvent,
    ModelMessagesTypeAdapter,
    ModelResponseStreamEvent,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.wrapper import WrapperModel


class BudgetExceeded(Exception):
    """A configured request or token budget has been exhausted."""


def write_json(path: Path, data: Any) -> None:
    """Atomically replace a report so interrupted writes do not corrupt it."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class Trace:
    """Append and flush events so failed trials retain their diagnostics."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, event: str, **data: Any) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"event": event, **data}, allow_nan=False) + "\n")


@dataclass
class TokenBudget:
    """Track observed usage across all trials in a run."""

    limit: int
    used: int = 0


@dataclass
class Measurements:
    """Measurements for one complete case, including follow-up tasks."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    executions: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class TimedStream:
    """Observe public stream events while leaving response assembly to the provider."""

    def __init__(self, stream: StreamedResponse, call: dict[str, Any], started: float) -> None:
        self.stream = stream
        self.call = call
        self.started = started

    async def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for event in self.stream:
            text = (isinstance(event, PartStartEvent) and isinstance(event.part, TextPart) and bool(event.part.content)) or (
                isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta) and bool(event.delta.content_delta)
            )
            if text and self.call["first_text_seconds"] is None:
                self.call["first_text_seconds"] = monotonic() - self.started
            yield event

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


class MeasuredModel(WrapperModel):
    """Wrap the actual provider without changing the worker's generation loop."""

    def __init__(
        self, wrapped: Any, *, trace: Trace, measurements: Measurements, budget: TokenBudget, max_calls: int, max_tokens: int
    ) -> None:
        super().__init__(wrapped)
        self.trace = trace
        self.measurements = measurements
        self.budget = budget
        self.max_calls = max_calls
        self.max_tokens = max_tokens

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        measurements = self.measurements
        if len(measurements.calls) >= self.max_calls:
            raise BudgetExceeded("Trial model-call limit reached")
        if measurements.input_tokens + measurements.output_tokens >= self.max_tokens:
            raise BudgetExceeded("Trial token limit reached")
        if self.budget.used >= self.budget.limit:
            raise BudgetExceeded("Run token limit reached")

        call: dict[str, Any] = {"first_text_seconds": None, "total_seconds": None, "usage": None, "error": None}
        measurements.calls.append(call)
        self.trace.write("model_request", messages=ModelMessagesTypeAdapter.dump_python(messages, mode="json"))
        started = monotonic()
        try:
            async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as stream:
                try:
                    yield cast(StreamedResponse, TimedStream(stream, call, started))
                finally:
                    call["total_seconds"] = monotonic() - started
                    response = stream.get()
                    usage = asdict(response.usage)
                    call["usage"] = usage
                    measurements.input_tokens += response.usage.input_tokens
                    measurements.output_tokens += response.usage.output_tokens
                    self.budget.used += response.usage.input_tokens + response.usage.output_tokens
                    self.trace.write("model_response", messages=ModelMessagesTypeAdapter.dump_python([response], mode="json"))
        except BaseException as error:
            call["error"] = type(error).__name__
            raise
        finally:
            if call["total_seconds"] is None:
                call["total_seconds"] = monotonic() - started
            self.trace.write("model_timing", **call)
        if measurements.input_tokens + measurements.output_tokens > self.max_tokens or self.budget.used > self.budget.limit:
            raise BudgetExceeded("Observed token usage exceeded the budget; generated code was not executed")


class MeasuredShell(RemoteShell):
    """Separate setup and grading executions from model-generated executions."""

    def __init__(self, client: Any, *, trace: Trace, measurements: Measurements) -> None:
        super().__init__(client)
        self.trace = trace
        self.measurements = measurements
        self.phase = "setup"

    async def stream(self, code: str, *, interactive: bool = False) -> Result:
        started = monotonic()
        self.trace.write("execution_request", phase=self.phase, code=code)
        result = None
        try:
            result = await super().stream(code, interactive=interactive)
            return result
        finally:
            execution = {
                "phase": self.phase,
                "seconds": monotonic() - started,
                "result": result.model_dump() if result is not None else None,
            }
            self.measurements.executions.append(execution)
            self.trace.write("execution_result", **execution)
