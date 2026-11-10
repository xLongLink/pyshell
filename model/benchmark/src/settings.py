from typing import Literal
from pathlib import Path
from pydantic import Field, BaseModel, ConfigDict, model_validator

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    """Configure evaluation data, output, and model inference."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    # Evaluation data
    data: str = Field(default="LongLink/pyshell-worker-data", min_length=1)
    data_revision: str | None = Field(default="1ed43ba23eff6438850921dca30665e5608fb8b7", pattern=r"^[0-9a-f]{40}$")

    # Run
    calls: int = Field(default=4, gt=0)
    output: Path = ROOT / "artifacts/results.jsonl"
    adapter: Path = ROOT.parent / "train/artifacts/candidate"

    # Inference
    device: Literal["cpu", "cuda"] = "cpu"
    tokens: int = Field(default=384, gt=0)
    threads: int = Field(default=16, gt=0)

    @model_validator(mode="after")
    def validate_data(self) -> "Settings":
        """Require a pinned revision for remotely loaded evaluation data."""
        if not Path(self.data).is_dir() and self.data_revision is None:
            raise ValueError("A remote benchmark requires a pinned revision")
        return self


SETTINGS = Settings()
