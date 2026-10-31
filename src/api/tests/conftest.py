import pytest


@pytest.fixture(autouse=True)
def execution_credentials(monkeypatch):
    monkeypatch.setenv("PYSHELL_EXECUTION_TOKEN", "test-token")
