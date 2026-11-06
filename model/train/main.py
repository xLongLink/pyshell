import sys

from model.train.src import settings, training

if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit(
            "Training takes no arguments; edit model/train/src/settings.py instead."
        )
    training.train(settings.SETTINGS)
