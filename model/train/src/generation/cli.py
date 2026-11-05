import argparse
from importlib import import_module

COMMANDS = {
    "join-curriculum": "Consolidate matched-only versus keep-all joins with skill replay",
    "binding-curriculum": "Build object-role binding, schema recovery and skill replay",
    "binding-evaluation": "Freeze container-verified binding evaluation before training",
    "binding": "Build execution-verified API and schema binding supervision",
    "executed": "Build balanced training trajectories using real container feedback",
    "transfer": "Build template-held-out contrastive tasks",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in COMMANDS.items():
        commands.add_parser(name, add_help=False, help=description)
    args, remaining = parser.parse_known_args(argv)
    module = args.command.replace("-", "_")
    return import_module(f"model.train.src.generation.{module}").main(remaining)
