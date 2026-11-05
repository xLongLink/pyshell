from dataclasses import dataclass

CONTRACT = (
    "Use the existing Python workspace, not invented input data. "
    "You may execute Python and inspect its output. np, pd, statistics, and json are available. "
    "Save the requested answer in result and call done(str(result)) when finished."
)


@dataclass
class ConverterSpec:
    gain: int
    offset: int
    multiplier: str
    shift: str


@dataclass
class TaskFixture:
    record: dict
    values: list[int]
    converter: ConverterSpec | None = None
    extra: int | None = None
