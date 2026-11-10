from typing import Literal
from pathlib import Path
from pydantic import Field, BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    """Configure evaluation data, output, and model inference."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    # Evaluation data
    data: str = Field(default=str(ROOT / "data"), min_length=1)

    # Run
    calls: int = Field(default=4, gt=0)
    output: Path = ROOT / "artifacts/results.jsonl"
    adapter: Path = ROOT.parent / "train/artifacts/candidate"

    # Inference
    device: Literal["cpu", "cuda"] = "cpu"
    tokens: int = Field(default=384, gt=0)
    threads: int = Field(default=16, gt=0)
    batch_size: int = Field(default=1, gt=0)


SETTINGS = Settings()
