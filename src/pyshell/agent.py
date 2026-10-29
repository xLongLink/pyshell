import sys
import json
import time
from .shell import PythonShell
from .streaming import stream_response

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
Import the modules you need before using them. No modules are pre-imported;
imports persist until the session is reset.
"""


class Agent:
    def __init__(self, client, model, service_tier="auto"):
        self.client = client
        self.model = model
        self.service_tier = service_tier
        self.reset()

    def reset(self):
        self.shell = PythonShell()
        self.previous_response_id = None
        self.pending_feedback = None

    def warm_up(self):
        started = time.perf_counter()
        print("Preparing API connection…", file=sys.stderr, flush=True)

        # Warm-up must not execute code or create stored conversation state.
        try:
            self.client.responses.create(
                model=self.model,
                service_tier=self.service_tier,
                reasoning={"effort": "low"},
                input="Reply with only 0.",
                max_output_tokens=16,
                store=False,
            )

        # A failed warm-up must not prevent interactive use.
        except Exception as exc:
            print(f"Warm-up failed: {exc}", file=sys.stderr, flush=True)
        else:
            duration = time.perf_counter() - started
            print(f"API warm-up completed in {duration:.3f}s.", file=sys.stderr, flush=True)

    def run(self, instruction):
        message = instruction
        while True:

            # Include feedback retained from a completed or interrupted turn.
            messages = []
            if self.pending_feedback is not None:
                messages.append({"role": "user", "content": self.pending_feedback})
            messages.append({"role": "user", "content": message})

            # Resume the stored response only after its stream completed.
            kwargs = {
                "model": self.model,
                "service_tier": self.service_tier,
                "reasoning": {"effort": "low"},
                "instructions": INSTRUCTIONS,
                "input": messages,
            }

            # A fresh conversation has no response ID to resume.
            if self.previous_response_id:
                kwargs["previous_response_id"] = self.previous_response_id

            # Executed lines survive failures, so retain their output for retry.
            try:
                response = stream_response(self.client, self.shell, kwargs)
            except (Exception, KeyboardInterrupt) as exc:
                result = self.shell.result()
                result["stream_error"] = f"{type(exc).__name__}: {exc}"

                # Older feedback remains pending until a response succeeds.
                feedback = json.dumps(result)
                if self.pending_feedback is not None:
                    feedback = self.pending_feedback + "\n" + feedback
                self.pending_feedback = feedback
                raise

            # Advance conversation state only after receiving usable code.
            if not response.output_text.strip():
                raise RuntimeError("The model returned no Python code.")
            self.pending_feedback = None
            self.previous_response_id = response.id
            result = self.shell.result()
            message = json.dumps(result)

            # Preserve the final result for the next user instruction.
            if result["done"]:
                self.pending_feedback = message
                return
