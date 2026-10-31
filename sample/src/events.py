from typing import Literal, Annotated
from pydantic import Field, BaseModel, ConfigDict, TypeAdapter


class Result(BaseModel):
    """Confirm a completed Python execution."""

    model_config = ConfigDict(extra="forbid", strict=True)

    # A result includes failures as data, alongside captured output.
    type: Literal["result"]
    stdout: str
    stderr: str
    exception: str | None
    done: bool


class Finished(BaseModel):
    """Mark a clean end to the execution stream."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["execution_finished"]


execution_event = TypeAdapter(Annotated[Result | Finished, Field(discriminator="type")])
