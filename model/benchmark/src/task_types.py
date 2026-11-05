import random
from dataclasses import dataclass

CONTRACT = (
    "Use the existing Python workspace, not invented input data. "
    "You may execute Python and inspect its output. np, pd, statistics, and json are available. "
    "Save the requested answer in result and call done(str(result)) when finished."
)


@dataclass
class TaskSeed:
    rng: random.Random
    name: str
    values: list[int]
    labels: list[str]

    @property
    def other(self):
        return self.name + "_other"

    @property
    def setup(self):
        return f"{self.name} = {self.values!r}"

    @property
    def inspection(self):
        return f"print(type({self.name}).__name__)\nprint({self.name})"


@dataclass
class ConstructedTask:
    setup: str
    inspect: str
    code: str
    expected: object
    check: str
    recovery_code: str | None = None
    followup: dict | None = None
    extra: int | None = None
