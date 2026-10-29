from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    openai_api_key: SecretStr = Field(min_length=1)
    openai_model: str = Field(default="gpt-6.1-sol", min_length=1)
    openai_service_tier: Literal["auto", "default", "flex", "priority"] = "auto"
