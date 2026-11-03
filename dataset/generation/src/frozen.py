"""Load historical generator snapshots without restoring obsolete entry points."""

import runpy
import sys

from dataset.benchmark.src import reference
from dataset.generation.src import proof


def load_generator(path):
    aliases = {
        "dataset.benchmark.cpu_tasks": proof,
        "model.train.cpu_curriculum": reference,
    }
    previous = {name: sys.modules.get(name) for name in aliases}
    try:
        sys.modules.update(aliases)
        return runpy.run_path(str(path), run_name="frozen_generator")
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
