import argparse
from importlib import import_module

COMMANDS = {
    "evaluate": ("evaluate", "Compare baseline and candidate on frozen development tasks"),
    "final": ("final", "Explicitly evaluate and independently verify the final holdout"),
    "verify": ("verification", "Independently replay a completed final evaluation"),
    "worker": ("worker", "Run the container-only execution protocol"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    for name, (_, description) in COMMANDS.items():
        commands.add_parser(
            name,
            add_help=name in {"evaluate", "final"},
            help=description,
            description=description + ". Configure model/benchmark/src/settings.py.",
        )
    args, remaining = parser.parse_known_args(argv)
    command = args.command or "evaluate"
    module, _ = COMMANDS[command]
    handler = import_module(f"model.benchmark.src.{module}")
    if command in {"evaluate", "final"}:
        if remaining:
            parser.error("Benchmark takes no flags; edit model/benchmark/src/settings.py")
        return handler.main()
    return handler.main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
