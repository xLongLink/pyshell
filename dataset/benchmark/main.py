"""Entry point for worker benchmarks and execution-proof evaluation."""

import argparse
from importlib import import_module

COMMANDS = {
    "list": ("cli", "List predefined worker cases"),
    "run": ("cli", "Run the provider-backed worker benchmark"),
    "report": ("cli", "Summarize saved trial records"),
    "cpu": ("cpu_evaluate", "Evaluate frozen CPU execution tasks"),
    "evaluate": ("evaluate", "Evaluate a local GGUF adapter"),
    "checkpoint": ("checkpoint_eval", "Export and evaluate a saved checkpoint"),
    "audit": ("audit", "Audit execution against dataset records"),
    "audit-pipeline": ("audit_pipeline", "Compare historical HF and GGUF checkpoints"),
    "syntax": ("syntax", "Check generated Python syntax"),
    "worker": ("cpu_worker", "Run the container-only execution protocol"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in COMMANDS.items():
        commands.add_parser(name, add_help=False, help=description)
    args, remaining = parser.parse_known_args(argv)
    module, _ = COMMANDS[args.command]
    if module == "cli":
        remaining = [args.command, *remaining]
    return import_module(f"dataset.benchmark.src.{module}").main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
