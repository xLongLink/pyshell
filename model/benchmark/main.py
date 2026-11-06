import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate one model. Configure model/benchmark/src/settings.py.")
    parser.parse_args(argv)
    from model.benchmark.src.evaluate import run
    from model.benchmark.src.settings import SETTINGS

    run(SETTINGS)


if __name__ == "__main__":
    main()
