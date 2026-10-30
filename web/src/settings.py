from typing import Literal
from pathlib import Path
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load agent configuration from the environment."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env", env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True
    )

    openai_api_key: SecretStr = Field(min_length=1)
    openai_model: str = Field(default="gpt-6.1-sol", min_length=1)
    openai_service_tier: Literal["auto", "default", "flex", "priority"] = "auto"


settings = Settings()
