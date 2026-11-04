"""Entry point for dataset generation, model training, and export."""

import argparse
from importlib import import_module

COMMANDS = {
    "generate": ("generation.cli", "Generate or convert training datasets"),
    "export": ("export", "Export a saved adapter to GGUF"),
    "checkpoint": ("checkpoint_eval", "Export and evaluate a saved checkpoint"),
    "audit-pipeline": ("audit_pipeline", "Compare historical HF and GGUF checkpoints"),
    "iterate": (
        "iteration",
        "Run sequential CPU pilots with development selection and final replay",
    ),
    "train": ("training", "Train or resume a LoRA adapter"),
    "experiment": ("experiment", "Run a development-gated execution-proof experiment"),
    "pipeline": ("pipeline", "Train and evaluate through the GGUF worker benchmark"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in COMMANDS.items():
        commands.add_parser(name, add_help=False, help=description)
    args, remaining = parser.parse_known_args(argv)
    module, _ = COMMANDS[args.command]
    return import_module(f"model.train.src.{module}").main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
