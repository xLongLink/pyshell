"""Entry point for dataset generation and conversion."""

import argparse
from importlib import import_module

COMMANDS = {
    "executed": (
        "executed",
        "Build balanced training trajectories using real container feedback",
    ),
    "proof": ("proof", "Build independent CPU execution tasks"),
    "transfer": ("transfer", "Build template-held-out contrastive tasks"),
    "curriculum": ("curriculum", "Build a training-only recovery curriculum"),
    "legacy": ("legacy", "Reproduce the original chat corpus"),
    "expanded": ("expanded", "Build expanded raw supervision"),
    "convert-chat": (
        "convert_chat",
        "Convert validated chat records to the raw protocol",
    ),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in COMMANDS.items():
        commands.add_parser(name, add_help=False, help=description)
    args, remaining = parser.parse_known_args(argv)
    module, _ = COMMANDS[args.command]
    return import_module(f"dataset.generation.src.{module}").main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
