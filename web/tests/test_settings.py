import os
import pytest
from pathlib import Path
from pydantic import ValidationError
from web.src.script import Settings


@pytest.fixture
def clean_environment(monkeypatch):
    for name in os.environ:
        monkeypatch.delenv(name)


def test_loads_dotenv_from_configured_path(clean_environment, tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=test-key\nOPENAI_MODEL=test-model\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(Settings.model_config, "env_file", tmp_path / ".env")

    settings = Settings()

    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert "test-key" not in repr(settings)
    assert settings.openai_model == "test-model"


def test_default_dotenv_is_in_web():
    from web.src import script

    assert Settings.model_config["env_file"] == Path(script.__file__).resolve().parents[1] / ".env"


def test_environment_overrides_dotenv(clean_environment, tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=file-key\nOPENAI_MODEL=file-model\n")
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")

    assert Settings(_env_file=dotenv).openai_model == "environment-model"


@pytest.mark.parametrize("values", [{}, {"openai_api_key": ""}])
def test_required_key(clean_environment, values):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)
