"""A terminal agent that communicates by executing Python."""

import sys
import argparse
from .agent import Agent
from pydantic import ValidationError
from .settings import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    try:
        settings = Settings()
    except ValidationError as exc:

        # Identify invalid settings without exposing their input values.
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']).upper()}: {error['msg']}"
            for error in exc.errors(include_input=False)
        )
        parser.error(f"Invalid configuration: {details}. Set values in .env or the environment.")

    # Keep --help available even when the API dependency is missing.
    try:
        from openai import OpenAI
    except ImportError:
        parser.error("Install dependencies with uv sync, then run uv run pyshell")

    # Configuration and client failures exit before opening the prompt.
    try:
        agent = Agent(
            OpenAI(api_key=settings.openai_api_key.get_secret_value()),
            settings.openai_model,
            settings.openai_service_tier,
        )

    # Initialization can fail before an agent exists.
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Error: {exc or 'interrupted'}", file=sys.stderr)
        return 1
    return _run_interactive(agent)


def _run_interactive(agent):
    print(f"Python agent shell ({agent.model}, tier={agent.service_tier}) "
          "— /exit to quit, /reset for fresh state")
    print("Warning: generated Python runs immediately with your full local permissions.")

    # Interrupting warm-up still leaves the session usable.
    try:
        agent.warm_up()
    except KeyboardInterrupt:
        print("\nWarm-up interrupted.", file=sys.stderr)

    # Ctrl-D exits; Ctrl-C at the prompt only cancels the current input.
    while True:
        try:
            instruction = input("\npyshell> ").strip()

        # End-of-input ends the session without sending another request.
        except EOFError:
            print()
            break

        # Cancelling prompt input does not reset the agent.
        except KeyboardInterrupt:
            print()
            continue

        # Shell commands never reach the model.
        if instruction in ("/exit", "/quit"):
            break
        if instruction == "/reset":

            # Report reset failure without discarding the previous session.
            try:
                agent.reset()

            # Failed resets do not terminate the prompt loop.
            except (Exception, KeyboardInterrupt) as exc:
                print(f"Reset failed; previous session kept: {exc or 'interrupted'}", file=sys.stderr)
            else:
                print("Python namespace and conversation reset (filesystem changes remain).")
            continue

        # Empty input and failed instructions must not close the prompt.
        if not instruction:
            continue
        try:
            agent.run(instruction)

        # An interrupted request may already have executed some Python.
        except KeyboardInterrupt:
            print("\nInterrupted.")

        # API and execution failures leave the session available for another turn.
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
    return 0
