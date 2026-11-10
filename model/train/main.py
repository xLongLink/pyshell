import sys

if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit(
            "Training takes no arguments; edit model/train/src/settings.py instead."
        )
    from model.train.src import settings, training

    training.train(settings.SETTINGS)
