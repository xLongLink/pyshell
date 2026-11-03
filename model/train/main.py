"""Entry point for model training, experiments, and independent verification."""

import argparse
from importlib import import_module

COMMANDS = {
    "train": ("training", "Train or resume a CPU LoRA adapter"),
    "experiment": ("experiment", "Run a development-gated execution-proof experiment"),
    "pipeline": ("pipeline", "Train and evaluate through the GGUF worker benchmark"),
    "verify": ("verification", "Independently replay a completed final evaluation"),
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
