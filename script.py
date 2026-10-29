"""A terminal agent that executes Python through the container API."""

import argparse
import json
import sys
from typing import Literal

import httpx
from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


INSTRUCTIONS = """You are an agent working in a persistent Python session inside a container.
Respond with ONLY executable Python code, without Markdown fences or prose.
Your complete response is executed through a local HTTP API. Use Python's
standard library, installed packages, and subprocess as necessary.
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
"""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True,
    )

    openai_api_key: SecretStr = Field(min_length=1)
    openai_model: str = Field(default="gpt-6.1-sol", min_length=1)
    openai_service_tier: Literal["auto", "default", "flex", "priority"] = "auto"
    pyshell_api_url: str = "http://127.0.0.1:8000"


class RemoteShell:
    def __init__(self, url="http://127.0.0.1:8000"):
        self.url = url.rstrip("/")

    def execute(self, code):
        try:
            response = httpx.post(f"{self.url}/", json={"code": code}, timeout=None, trust_env=False)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Execution request failed; code may already have run. "
                "Inspect the session or use /reset before retrying."
            ) from exc
        return response.json()

    def reset(self):
        response = httpx.post(f"{self.url}/reset", timeout=None, trust_env=False)
        response.raise_for_status()


def stream_response(client, kwargs):
    print()
    try:
        with client.responses.stream(**kwargs) as stream:
            for event in stream:
                if event.type != "response.output_text.delta":
                    continue
                print(event.delta, end="", flush=True)
            response = stream.get_final_response()
            if response.status != "completed":
                raise RuntimeError(f"Model response {response.status}; code was not executed.")
            return response
    finally:
        print(flush=True)


class Agent:
    def __init__(self, client, model, service_tier="auto", shell=None):
        self.client = client
        self.model = model
        self.service_tier = service_tier
        self.shell = shell if shell is not None else RemoteShell()
        self.previous_response_id = None
        self.pending_feedback = None

    def reset(self):
        self.shell.reset()
        self.previous_response_id = None
        self.pending_feedback = None

    def run(self, instruction):
        message = instruction
        while True:
            messages = []
            if self.pending_feedback is not None:
                messages.append({"role": "user", "content": self.pending_feedback})
            messages.append({"role": "user", "content": message})
            kwargs = {
                "model": self.model, "service_tier": self.service_tier,
                "reasoning": {"effort": "low"}, "instructions": INSTRUCTIONS, "input": messages,
            }
            if self.previous_response_id:
                kwargs["previous_response_id"] = self.previous_response_id
            try:
                response = stream_response(self.client, kwargs)
            except (Exception, KeyboardInterrupt) as exc:
                feedback = json.dumps({"stream_error": f"{type(exc).__name__}: {exc}", "executed": False})
                if self.pending_feedback is not None:
                    feedback = self.pending_feedback + "\n" + feedback
                self.pending_feedback = feedback
                raise
            if not response.output_text.strip():
                raise RuntimeError("The model returned no Python code.")
            result = self.shell.execute(response.output_text)
            print(result["stdout"], end="", flush=True)
            print(result["stderr"], end="", file=sys.stderr, flush=True)
            if result["exception"]:
                print(result["exception"], end="", file=sys.stderr, flush=True)
            self.pending_feedback = None
            self.previous_response_id = response.id
            message = json.dumps(result)
            if result["done"]:
                self.pending_feedback = message
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
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
        parser.error("Install dependencies with uv sync, then run uv run python script.py")
    try:
        agent = Agent(
            OpenAI(api_key=settings.openai_api_key.get_secret_value()),
            settings.openai_model, settings.openai_service_tier,
            shell=RemoteShell(settings.pyshell_api_url),
        )
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Error: {exc or 'interrupted'}", file=sys.stderr)
        return 1
    return _run_interactive(agent)


def _run_interactive(agent):
    print(f"Python agent shell ({agent.model}, tier={agent.service_tier}) "
          "— Ctrl-C to quit, /reset for fresh state")
    print("Warning: generated Python runs without approval in the execution API's container.")
    while True:
        try:
            instruction = input("\npyshell> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if instruction == "/reset":
            try:
                agent.reset()
            except KeyboardInterrupt:
                print("\nExiting client; the server may still be processing the reset.")
                break
            except Exception as exc:
                print(f"Reset failed; previous session kept: {exc or 'interrupted'}", file=sys.stderr)
            else:
                print("Python namespace and conversation reset (filesystem changes remain).")
            continue
        if not instruction:
            continue
        try:
            agent.run(instruction)
        except KeyboardInterrupt:
            print("\nInterrupted locally; any submitted Python may still be running on the server.")
            break
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
