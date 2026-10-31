from typing import Literal, Annotated
from pydantic import Field, BaseModel, ConfigDict, TypeAdapter


class Event(BaseModel):
    """Validate execution events at the container boundary."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Result(Event):
    """Confirm a completed Python execution."""

    # A result includes failures as data, alongside captured output.
    type: Literal["result"]
    stdout: str
    stderr: str
    exception: str | None
    done: bool


class Finished(Event):
    """Mark a clean end to the execution stream."""

    type: Literal["execution_finished"]


execution_event = TypeAdapter(Annotated[Result | Finished, Field(discriminator="type")])
