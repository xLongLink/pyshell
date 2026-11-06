from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    # Frozen evaluation data
    data: str = Field(default="LongLink/pyshell-worker-data", min_length=1)
    prefix: str = Field(default="experiments/executed-transfer-v1/evaluation", min_length=1)
    data_revision: str | None = Field(default="1ed43ba23eff6438850921dca30665e5608fb8b7", pattern=r"^[0-9a-f]{40}$")

    # Artifacts
    output: Path = ROOT / "artifacts/comparison"
    baseline: Path = ROOT.parent / "train/weights"
    candidate: Path = ROOT.parent / "train/artifacts/candidate"

    # Comparable inference
    device: Literal["cpu", "cuda"] = "cpu"
    tokens: int = Field(default=384, gt=0)
    threads: int = Field(default=16, gt=0)

    @model_validator(mode="after")
    def validate_data(self):
        if not Path(self.data).is_dir() and self.data_revision is None:
            raise ValueError("A remote benchmark requires a pinned revision")
        return self


SETTINGS = Settings()
