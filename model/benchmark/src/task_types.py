"""Task records shared by fixture builders and supervision generators."""

from dataclasses import dataclass

FAMILIES = (
    "statistics",
    "grouped_revenue",
    "monthly_totals",
    "missing_values",
    "left_join",
    "inspect_callable",
    "recover_execution",
    "stateful_followup",
    "missing_input",
    "linear_system",
    "filtered_means",
    "joined_totals",
)
CONTRACT = (
    "Use the existing Python workspace, not invented input data. "
    "You may execute Python and inspect its output. np, pd, statistics, and json are available. "
    "Save the requested answer in result and call done(str(result)) when finished."
)


@dataclass
class TaskRecord:
    id: str
    family: str
    setup: str
    instruction: str
    inspect: str
    code: str
    check: str
    followup: dict | None


@dataclass
class ConverterSpec:
    gain: int
    offset: int
    multiplier: str
    shift: str


@dataclass
class TaskFixture:
    record: TaskRecord
    values: list[int]
    converter: ConverterSpec | None = None
    extra: int | None = None
