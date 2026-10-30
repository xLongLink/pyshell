import json
import asyncio
from typing import Literal
from pydantic_ai import Agent as PydanticAgent
from web.src.shell import RemoteShell
from collections.abc import Callable
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai.models import Model
from pydantic_ai.messages import ModelMessage

INSTRUCTIONS = """You are an agent working in a persistent Python session inside a container.
Respond with ONLY executable Python code, without Markdown fences or prose.
Your complete response is executed through a local HTTP API. Use Python's
standard library, installed packages, and subprocess as necessary.
Use the provided ui.plot for plotting, not matplotlib or other plotting libraries.
Variables, imports, and working-directory changes persist across executions and
user turns. Write normal Python scripts, without interactive prompt text.
Use print() to display results; bare expressions are not displayed automatically.
Execution begins only after your entire response is received successfully.
After each response you receive stdout, stderr, and any exception as JSON.
Use the results to continue or correct errors. Never invent execution results.
When the user's task is complete, print any useful final result and call done().
done() is a provided function, not something you should define or overwrite.
Do not call input(): the user supplies instructions through the web interface.
Do not perform destructive operations unless the user requested them.
Import the modules you need before using them. No modules are pre-imported;
imports persist until the session is reset.

The provided ui object publishes rich results through A2UI. It is available
without imports. These are normal synchronous Python calls; do not await them.
Use ui.plot(data, layout={}, surface_id=None), a restricted Plotly-style figure:
- data is a list of traces. Each trace requires a type discriminator.
- {"type": "scatter", "name": "Trend", "x": [0, 5, 10], "y": [1, 4, 8], "mode": "lines"}
  supports mode "lines" (default), "markers", or "lines+markers".
- {"type": "bar", "name": "Sales", "x": ["A", "B"], "y": [42, 50]} compares values.
- {"type": "pie", "labels": ["A", "B"], "values": [42, 50]} shows nonnegative shares.
  A pie figure must contain exactly one trace. Scatter and bar traces can mix.
- Paired arrays must have equal lengths. All x values in a figure must be numeric
  or all strings (categories); numeric x values are spaced numerically.
  Coordinates remain in input order. Use None for missing y/pie values; lines
  preserve gaps and pie omits missing slices. Each trace needs at least one point.
- At most 20 traces and 200 total points per figure.
- layout supports {"title": {"text": "Revenue"}, "xaxis": {"title": {"text": "Time"}},
  "yaxis": {"title": {"text": "Amount"}}, "showlegend": True}. All are optional.
  Other Plotly options and ui.chart.* constructors are unsupported.
- ui.remove(surface_id): remove an existing surface.
Each creation call returns a widget object, not a string. Keep it in a variable:
chart = ui.plot([{"type": "scatter", "x": ["Jan"], "y": [42]}], layout={"title": {"text": "Revenue"}})
chart.update(data=[{"type": "scatter", "x": ["Jan"], "y": [50]}]) publishes an explicit update.
Batch data and layout changes in one update() call. Supplied top-level fields
replace their previous values, including all nested contents; omitted fields stay
unchanged. Invalid updates publish nothing and leave the widget unchanged.
New widgets appear in the center, which holds up to four widgets. A fifth moves
the oldest centered widget to the sidebar. chart.focus() moves a widget into the
center; chart.pin() or chart.unfocus() moves it to the sidebar. chart.unpin()
moves it back into the center. These placement operations are backend-only.
chart.remove() removes its surface. chart.surface_id is its stable identifier.
All widgets share these methods. Removed or replaced widget handles are invalid.
Widget attributes are read-only outside update(); identity and type cannot change.
Reuse a widget or a short named surface_id to avoid duplication. Surfaces persist until
reset or removal; at most 20 may exist. Do not repeat successful mutations after
errors: earlier Python changes and published widgets may already have applied.
Each figure occupies its own surface. Text, metrics, tables, and layout widgets
are not available. Do not call ui.text(), ui.metric(), or ui.table(). Executable
UI, URLs, images, inputs, and arbitrary actions are not supported.
Convert pandas/numpy objects to plain Python values.
"""


async def stream_response(
    agent: PydanticAgent[None, str],
    message: str,
    message_history: list[ModelMessage],
    *,
    usage: RunUsage | None = None,
    usage_limits: UsageLimits | None = None,
) -> tuple[str, list[ModelMessage]]:
    """Consume the model stream and return only a completed response."""

    async with agent.run_stream(message, message_history=message_history, usage=usage, usage_limits=usage_limits) as stream:
        # Truncated responses must never reach execution.
        code = await stream.get_output()
        if stream.response.finish_reason != "stop":
            raise RuntimeError(f"Model response {stream.response.finish_reason}; code was not executed.")
        return code, stream.all_messages()


class Agent:
    """Run generated Python and feed results back to the model."""

    def __init__(
        self, model: Model, service_tier: Literal["auto", "default", "flex", "priority"] = "auto", shell: RemoteShell | None = None
    ) -> None:
        """Create a model-backed agent for a remote Python session."""
        self.shell = shell if shell is not None else RemoteShell()
        self._libraries: list[dict[str, str]] | None = None

        # Resolve the execution environment's package inventory on the first turn.
        self.agent = PydanticAgent(
            model, instructions=self._instructions, model_settings={"openai_service_tier": service_tier, "openai_reasoning_effort": "low"}
        )

        # Conversation state is committed only after execution succeeds.
        self.message_history = []
        self.pending_feedback = None

    def _instructions(self) -> str:
        if self._libraries is None:
            raise RuntimeError("Execution library inventory has not been loaded")
        return (
            INSTRUCTIONS
            + "\nInstalled packages in the execution environment (distribution names may differ from import names):\n"
            + json.dumps(self._libraries)
        )

    async def arun(self, instruction: str, on_event: Callable[[dict[str, object]], None] | None = None) -> str:
        """Generate complete code and consume live execution feedback."""
        if self.shell.client is None:
            raise RuntimeError("Execution transport is closed")
        if self._libraries is None:
            self._libraries = await self.shell.inventory()
        emit = on_event if on_event is not None else lambda _event: None
        usage = RunUsage()
        limits = UsageLimits(request_limit=50, total_tokens_limit=200_000)

        # Share usage limits across every execution step in this task.
        outputs = []
        message = instruction
        for _ in range(10):
            prompt = message
            if self.pending_feedback is not None:
                prompt = self.pending_feedback + "\n" + message

            # Preserve execution feedback across failed model requests.
            try:
                emit({"type": "code_start"})

                # Finish generation before executing any code.
                code, history = await stream_response(self.agent, prompt, self.message_history, usage=usage, usage_limits=limits)

            # Record that the interrupted response never reached the container.
            except (Exception, KeyboardInterrupt, asyncio.CancelledError) as error:
                feedback = json.dumps({"stream_error": f"{type(error).__name__}: {error}", "executed": False})

                # Keep prior execution feedback alongside this unexecuted request.
                if self.pending_feedback is not None:
                    feedback = self.pending_feedback + "\n" + feedback
                self.pending_feedback = feedback
                raise

            # Empty output is not a completed execution step.
            if not code.strip():
                raise RuntimeError("The model returned no Python code.")

            # Execute once; an ambiguous transport failure must not trigger a retry.
            emit({"type": "execution_start"})

            try:
                result = await self.shell.stream(code, emit)

            # Preserve history without allowing an ambiguous execution to replay.
            except BaseException:
                self.message_history = history
                self.pending_feedback = json.dumps(
                    {"execution_error": "Execution was not confirmed; earlier changes may have applied. Do not replay."}
                )
                raise

            # Retain the completed exchange before deciding whether to continue.
            self.pending_feedback = None
            self.message_history = history
            message = json.dumps(result)
            outputs.append(result["stdout"] + result["stderr"] + (result["exception"] or ""))

            # Carry final execution feedback into the next user turn.
            if result["done"]:
                self.pending_feedback = message
                return "\n".join(outputs) or "Task completed without printed output."
        self.pending_feedback = (
            message + "\nTask stopped at the ten-execution limit; completion was not confirmed. Do not replay earlier mutations."
        )
        raise RuntimeError("Task reached the ten-execution limit; earlier changes may have applied.")
