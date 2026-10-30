from pydantic import BaseModel, ConfigDict


class UIModel(BaseModel):
    """Validate immutable UI specifications."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
