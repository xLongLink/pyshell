from typing import Literal, Annotated
from pydantic import Field, BaseModel, JsonValue, ConfigDict, TypeAdapter


class Event(BaseModel):
    """Validate execution events at the container boundary."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Content(Event):
    """Carry validated UI protocol messages."""

    type: Literal["a2ui"]

    # Protocol messages retain their nested JSON structure.
    messages: list[dict[str, JsonValue]]


class WidgetCommand(Event):
    """Address one existing widget's placement."""

    type: Literal["widget"]
    action: Literal["focus", "pin"]
    surface_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")


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


execution_event = TypeAdapter(Annotated[Content | WidgetCommand | Result | Finished, Field(discriminator="type")])
