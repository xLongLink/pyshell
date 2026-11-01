from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load agent configuration from the environment."""

    model_config = SettingsConfigDict(extra="ignore", hide_input_in_errors=True)

    openai_api_key: SecretStr = Field(min_length=1)
    openai_live_model: str = Field(default="gpt-live-1", min_length=1)
    openai_voice_backend_model: str = Field(default="gpt-6.1-sol", min_length=1)
