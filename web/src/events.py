from typing import Literal, Annotated
from pydantic import Field, BaseModel, JsonValue, ConfigDict, TypeAdapter


class Event(BaseModel):
    """Validate execution events at the container boundary."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Output(Event):
    """Carry a fragment of Python output."""

    type: Literal["stdout_delta", "stderr_delta"]
    text: str


class Content(Event):
    """Carry validated UI protocol messages."""

    type: Literal["a2ui"]
    messages: list[dict[str, JsonValue]]


class WidgetCommand(Event):
    """Address one existing widget's placement."""

    type: Literal["widget"]
    action: Literal["focus", "unfocus", "pin", "unpin"]
    surface_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")


class Result(Event):
    """Confirm a completed Python execution."""

    type: Literal["result"]
    stdout: str
    stderr: str
    exception: str | None
    done: bool


class Finished(Event):
    """Mark a clean end to the execution stream."""

    type: Literal["execution_finished"]


execution_event = TypeAdapter(Annotated[Output | Content | WidgetCommand | Result | Finished, Field(discriminator="type")])
