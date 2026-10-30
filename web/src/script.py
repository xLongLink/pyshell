"""A terminal agent that executes Python through the container API."""

import sys
import json
import httpx
import argparse
from typing import Literal, TypedDict
from pathlib import Path
from pydantic import Field, SecretStr, ValidationError
from pydantic_ai import Agent as PydanticAgent
from collections.abc import Callable
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_ai.models import Model
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

INSTRUCTIONS = """You are an agent working in a persistent Python session inside a container.
Respond with ONLY executable Python code, without Markdown fences or prose.
Your complete response is executed through a local HTTP API. Use Python's
standard library, installed packages, and subprocess as necessary.
pandas, numpy, openpyxl, and scipy are installed for data analysis and Excel files.
Use the provided ui.plot for plotting, not matplotlib or other plotting libraries.
Variables, imports, and working-directory changes persist across executions and
user turns. Write normal Python scripts, without interactive prompt text.
Use print() to display results; bare expressions are not displayed automatically.
Execution begins only after your entire response is received successfully.
After each response you receive stdout, stderr, and any exception as JSON.
Use the results to continue or correct errors. Never invent execution results.
When the user's task is complete, print any useful final result and call done().
done() is a provided function, not something you should define or overwrite.
Do not call input(): the user supplies instructions through the outer shell.
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


class Settings(BaseSettings):
    """Load terminal client configuration from the environment."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env", env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True
    )

    # Environment variables override the client-local dotenv file.
    openai_api_key: SecretStr = Field(min_length=1)
    openai_model: str = Field(default="gpt-6.1-sol", min_length=1)
    openai_service_tier: Literal["auto", "default", "flex", "priority"] = "auto"
    pyshell_api_url: str = "http://127.0.0.1:8000"


class ExecutionResult(TypedDict):
    """Output returned by the execution API."""

    stdout: str
    stderr: str
    exception: str | None
    done: bool
    ui: list[dict[str, object]]


# Requests bypass proxy settings for the local execution server.
class RemoteShell:
    """Access the container's persistent Python session."""

    def __init__(self, url: str = "http://127.0.0.1:8000") -> None:
        """Select the execution API endpoint."""
        self.url = url.rstrip("/")

    def execute(self, code: str) -> ExecutionResult:
        """Submit code once and wait for its result."""
        try:
            response = httpx.post(f"{self.url}/", json={"code": code}, timeout=None, trust_env=False)
            response.raise_for_status()

        # Transport failure does not mean the server skipped execution.
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Execution request failed; code may already have run. Inspect the session or use /reset before retrying."
            ) from exc
        return response.json()

    def reset(self) -> None:
        """Clear the remote namespace after any active execution finishes."""
        response = httpx.post(f"{self.url}/reset", timeout=None, trust_env=False)
        response.raise_for_status()


def stream_response(
    agent: PydanticAgent[None, str], message: str, message_history: list[ModelMessage], on_code: Callable[[str], None] | None = None
) -> tuple[str, list[ModelMessage]]:
    """Display generated code and return only a completed response."""
    print()
    try:
        stream = agent.run_stream_sync(message, message_history=message_history)
        for text in stream.stream_text(delta=True, debounce_by=None):
            print(text, end="", flush=True)
            if on_code is not None:
                on_code(text)

        # Display partial output, but never execute an incomplete response.
        code = stream.get_output()
        if stream.response.finish_reason != "stop":
            raise RuntimeError(f"Model response {stream.response.finish_reason}; code was not executed.")
        return code, stream.all_messages()
    finally:
        print(flush=True)


class Agent:
    """Run generated Python and feed results back to the model."""

    def __init__(
        self, model: Model, service_tier: Literal["auto", "default", "flex", "priority"] = "auto", shell: RemoteShell | None = None
    ) -> None:
        """Create a model-backed agent for a remote Python session."""
        self.model = model.model_name
        self.service_tier = service_tier
        self.agent = PydanticAgent(
            model, instructions=INSTRUCTIONS, model_settings={"openai_service_tier": service_tier, "openai_reasoning_effort": "low"}
        )
        self.shell = shell if shell is not None else RemoteShell()

        # Conversation state is committed only after execution succeeds.
        self.message_history = []
        self.pending_feedback = None

    def reset(self) -> None:
        """Clear remote state and conversation after a successful reset."""
        self.shell.reset()
        self.message_history = []
        self.pending_feedback = None

    def run(self, instruction: str, on_event: Callable[[dict[str, object]], None] | None = None) -> None:
        """Execute model responses until the task signals completion."""
        message = instruction
        while True:
            prompt = message
            if self.pending_feedback is not None:
                prompt = self.pending_feedback + "\n" + message

            # Preserve execution feedback across failed model requests.
            try:
                if on_event is not None:
                    on_event({"type": "code_start"})
                code, history = stream_response(
                    self.agent,
                    prompt,
                    self.message_history,
                    on_code=(lambda text: on_event({"type": "code_delta", "text": text})) if on_event is not None else None,
                )

            except (Exception, KeyboardInterrupt) as exc:
                feedback = json.dumps({"stream_error": f"{type(exc).__name__}: {exc}", "executed": False})

                # Keep prior execution feedback alongside this unexecuted request.
                if self.pending_feedback is not None:
                    feedback = self.pending_feedback + "\n" + feedback
                self.pending_feedback = feedback
                raise

            # Empty output is not a completed execution step.
            if not code.strip():
                raise RuntimeError("The model returned no Python code.")

            # Execute once; an ambiguous transport failure must not trigger a retry.
            if on_event is not None:
                on_event({"type": "execution_start"})
            result = self.shell.execute(code)
            print(result["stdout"], end="", flush=True)
            print(result["stderr"], end="", file=sys.stderr, flush=True)
            if result["exception"]:
                print(result["exception"], end="", file=sys.stderr, flush=True)

            # Retain the completed exchange before deciding whether to continue.
            self.pending_feedback = None
            self.message_history = history
            message = json.dumps(result)
            if on_event is not None:
                on_event({"type": "result", **{key: value for key, value in result.items() if key != "ui"}})
                if result.get("ui"):
                    messages = []
                    for item in result["ui"]:
                        if item.get("type") == "widget":
                            if messages:
                                on_event({"type": "a2ui", "messages": messages})
                                messages = []
                            on_event(item)
                        else:
                            messages.append(item)
                    if messages:
                        on_event({"type": "a2ui", "messages": messages})
            if result["done"]:
                self.pending_feedback = message
                return


def main() -> int:
    """Load configuration and start the terminal client."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        settings = Settings()

    # Report field errors without exposing secret configuration values.
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']).upper()}: {error['msg']}" for error in exc.errors(include_input=False)
        )
        parser.error(f"Invalid configuration: {details}. Set values in web/.env or the environment.")

    # Construct the model only after configuration has been validated.
    try:
        agent = Agent(
            OpenAIResponsesModel(settings.openai_model, provider=OpenAIProvider(api_key=settings.openai_api_key.get_secret_value())),
            settings.openai_service_tier,
            shell=RemoteShell(settings.pyshell_api_url),
        )

    # Initialization failure must not enter the interactive loop.
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Error: {exc or 'interrupted'}", file=sys.stderr)
        return 1
    return _run_interactive(agent)


def _run_interactive(agent: Agent) -> int:
    print(f"Python agent shell ({agent.model}, tier={agent.service_tier}) — Ctrl-C to quit, /reset for fresh state")
    print("Warning: generated Python runs without approval in the execution API's container.")

    while True:
        try:
            instruction = input("\npyshell> ").strip()

        # Local interruption exits the client, not the remote execution.
        except (EOFError, KeyboardInterrupt):
            print()
            break

        # Keep the conversation if resetting the remote session fails.
        if instruction == "/reset":
            try:
                agent.reset()

            # Reset may still be running remotely after local interruption.
            except KeyboardInterrupt:
                print("\nExiting client; the server may still be processing the reset.")
                break

            # Only a successful reset clears local conversation state.
            except Exception as exc:
                print(f"Reset failed; previous session kept: {exc or 'interrupted'}", file=sys.stderr)
            else:
                print("Python namespace and conversation reset (filesystem changes remain).")
            continue

        # Blank input must not issue a model request.
        if not instruction:
            continue
        try:
            agent.run(instruction)

        # Submitted code can outlive the client connection.
        except KeyboardInterrupt:
            print("\nInterrupted locally; any submitted Python may still be running on the server.")
            break

        # Non-interrupt errors leave the client available for another instruction.
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
