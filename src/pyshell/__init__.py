"""A terminal agent that communicates by executing Python."""

import argparse
import code
import contextlib
import io
import json
import sys
import time

from pydantic import ValidationError

from .settings import Settings


INSTRUCTIONS = """You are an agent working in a persistent local Python shell.
Respond with ONLY executable Python code, without Markdown fences or prose.
Your code executes immediately with the user's local permissions. Use Python's
standard library, installed packages, and subprocess as necessary.
Variables, imports, and working-directory changes persist across executions and
user turns. Each newline submits a line to a Python interactive console, and
expressions display their values automatically. Include a blank line to finish
each compound statement (for, if, def, class, try, etc.) before the next statement,
exactly as if typing at Python's >>> and ... prompts. Do not include prompt text.
Execution happens while you generate code, not after your response finishes.
After each response you receive stdout, stderr, and any exception as JSON.
Use the results to continue or correct errors. Never invent execution results.
When the user's task is complete, print any useful final result and call done().
done() is a provided function, not something you should define or overwrite.
Do not call input(): the user supplies instructions through the outer shell.
Do not perform destructive operations unless the user requested them.
"""


class Tee(io.TextIOBase):
    def __init__(self, terminal):
        self.terminal = terminal
        self.buffered = io.StringIO()

    def write(self, text):
        self.buffered.write(text)
        self.terminal.write(text)
        self.terminal.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()


class PythonShell(code.InteractiveConsole):
    def __init__(self):
        self.finished = False
        self.namespace = {"__name__": "__main__", "done": self.done}
        super().__init__(locals=self.namespace, filename="<agent>")
        self.begin_response()

    def begin_response(self):
        self.finished = False
        self.stdout = []
        self.stderr = []
        self.errors = []
        self.resetbuffer()

    def done(self):
        self.finished = True

    def write(self, text):
        self.errors.append(text)
        super().write(text)

    def runcode(self, compiled):
        try:
            exec(compiled, self.locals)
        except (Exception, SystemExit):
            self.finished = False
            self.showtraceback()

    def execute_line(self, line):
        stdout, stderr = Tee(sys.stdout), Tee(sys.stderr)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                return self.push(line)
        finally:
            self.stdout.append(stdout.buffered.getvalue())
            self.stderr.append(stderr.buffered.getvalue())

    def result(self):
        return {
            "stdout": "".join(self.stdout),
            "stderr": "".join(self.stderr),
            "exception": "".join(self.errors) or None,
            "done": self.finished,
        }


class Agent:
    def __init__(self, client, model, service_tier="auto"):
        self.client = client
        self.model = model
        self.service_tier = service_tier
        self.shell = PythonShell()
        self.previous_response_id = None
        self.pending_feedback = None

    def warm_up(self):
        started = time.perf_counter()
        print("Preparing API connection…", file=sys.stderr, flush=True)
        try:
            self.client.responses.create(
                model=self.model,
                service_tier=self.service_tier,
                reasoning={"effort": "low"},
                input="Reply with only 0.",
                max_output_tokens=16,
                store=False,
            )
        except Exception as exc:
            print(f"Warm-up failed: {exc}", file=sys.stderr, flush=True)
        else:
            duration = time.perf_counter() - started
            print(f"API warm-up completed in {duration:.3f}s.", file=sys.stderr, flush=True)

    def run(self, instruction):
        message = instruction
        while True:
            messages = []
            if self.pending_feedback is not None:
                messages.append({"role": "user", "content": self.pending_feedback})
            messages.append({"role": "user", "content": message})
            kwargs = {
                "model": self.model,
                "service_tier": self.service_tier,
                "reasoning": {"effort": "low"},
                "instructions": INSTRUCTIONS,
                "input": messages,
            }
            if self.previous_response_id:
                kwargs["previous_response_id"] = self.previous_response_id
            try:
                response = self._stream_response(kwargs)
            except (Exception, KeyboardInterrupt) as exc:
                result = self.shell.result()
                result["stream_error"] = f"{type(exc).__name__}: {exc}"
                feedback = json.dumps(result)
                if self.pending_feedback is not None:
                    feedback = self.pending_feedback + "\n" + feedback
                self._record_feedback(feedback)
                raise
            if not response.output_text.strip():
                raise RuntimeError("The model returned no Python code.")
            self.pending_feedback = None
            self.previous_response_id = response.id
            result = self.shell.result()
            message = json.dumps(result)
            if result["done"]:
                self._record_feedback(message)
                return

    def _stream_response(self, kwargs):
        self.shell.begin_response()
        started = time.perf_counter()
        first_token = None
        line = ""
        at_line_start = True
        wrote_text = False
        print()
        try:
            with self.client.responses.stream(**kwargs) as stream:
                for event in stream:
                    if event.type != "response.output_text.delta":
                        continue
                    if event.delta and first_token is None:
                        first_token = time.perf_counter() - started
                    for part in event.delta.splitlines(keepends=True):
                        if at_line_start:
                            print("... " if self.shell.buffer else ">>> ", end="", flush=True)
                        print(part, end="", flush=True)
                        at_line_start = part.endswith("\n")
                        wrote_text = True
                        line += part
                        if at_line_start:
                            self.shell.execute_line(line.removesuffix("\n").removesuffix("\r"))
                            line = ""
                response = stream.get_final_response()
                if response.status != "completed":
                    raise RuntimeError(
                        f"Model response {response.status}; remaining input discarded. "
                        "Previously executed lines remain in effect."
                    )
                if line:
                    print(flush=True)
                    at_line_start = True
                    self.shell.execute_line(line)
                if self.shell.buffer:
                    print("... ", flush=True)
                    self.shell.execute_line("")
                if self.shell.buffer:
                    self.shell.errors.append("Incomplete Python input discarded at end of response.")
                return response
        finally:
            self.shell.resetbuffer()
            if wrote_text and not at_line_start:
                print(flush=True)
            duration = time.perf_counter() - started
            ttft = f"{first_token:.3f}s" if first_token is not None else "n/a"
            print(f"[Time to first token: {ttft} | Total duration: {duration:.3f}s]",
                  file=sys.stderr, flush=True)

    def _record_feedback(self, message):
        # Keep even the final execution result in the conversation for the next turn.
        self.pending_feedback = message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction", nargs="*", help="Run a single instruction and exit")
    args = parser.parse_args()
    try:
        settings = Settings()
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']).upper()}: {error['msg']}"
            for error in exc.errors(include_input=False)
        )
        parser.error(f"Invalid configuration: {details}. Set values in .env or the environment.")
    try:
        from openai import OpenAI
    except ImportError:
        parser.error("Install dependencies with uv sync, then run uv run pyshell")
    agent = Agent(
        OpenAI(api_key=settings.openai_api_key.get_secret_value()),
        settings.openai_model,
        settings.openai_service_tier,
    )
    if args.instruction:
        try:
            agent.run(" ".join(args.instruction))
        except (Exception, KeyboardInterrupt) as exc:
            print(f"Error: {exc or 'interrupted'}", file=sys.stderr)
            return 1
        return 0
    print(f"Python agent shell ({settings.openai_model}, tier={settings.openai_service_tier}) "
          "— /exit to quit, /reset for fresh state")
    print("Warning: generated Python runs immediately with your full local permissions.")
    try:
        agent.warm_up()
    except KeyboardInterrupt:
        print("\nWarm-up interrupted.", file=sys.stderr)
    while True:
        try:
            instruction = input("\npyshell> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue
        if instruction in ("/exit", "/quit"):
            break
        if instruction == "/reset":
            agent = Agent(agent.client, settings.openai_model, settings.openai_service_tier)
            print("Python namespace and conversation reset (filesystem changes remain).")
            continue
        if not instruction:
            continue
        try:
            agent.run(instruction)
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
