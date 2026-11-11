from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, validate_default=True, allow_inf_nan=False
    )

    # Model and dataset
    base: str = Field(default="Qwen/Qwen3.5-0.8B", min_length=1)
    data: str = Field(default=str(ROOT / "data"), min_length=1)
    base_revision: str | None = Field(
        default="2fc06364715b967f1860aea9cf38778875588b17", pattern=r"^[0-9a-f]{40}$"
    )

    # Run
    cpu: bool = True
    seed: int = Field(default=713, ge=0)
    steps: int = Field(default=1250, gt=0)
    output: Path = ROOT / "artifacts/candidate"

    # Training
    dtype: Literal["float32", "bfloat16"] = "float32"
    batch_size: int = Field(default=1, gt=0)
    max_length: int = Field(default=2048, gt=0)
    learning_rate: float = Field(default=5e-5, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)

    # LoRA
    rank: int = Field(default=8, gt=0)
    alpha: int = Field(default=16, gt=0)
    target_modules: str = Field(
        default=r".*language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|out_proj))",
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_sources(self):
        if not Path(self.base).is_dir() and self.base_revision is None:
            raise ValueError("A remote base model requires a pinned revision")
        return self


SETTINGS = Settings()
