import httpx
import pytest

from script import RemoteShell


def test_remote_execution_and_reset(remote_shell):
    assert remote_shell.execute("value = 42")["exception"] is None
    assert remote_shell.execute("print(value)")["stdout"] == "42\n"
    remote_shell.reset()
    assert "NameError" in remote_shell.execute("print(value)")["exception"]


def test_execution_transport_failure_is_not_retried(monkeypatch):
    calls = []

    def post(*args, **kwargs):
        calls.append(args)
        raise httpx.ReadError("disconnected")

    monkeypatch.setattr("script.httpx.post", post)
    with pytest.raises(RuntimeError, match="may already have run"):
        RemoteShell().execute("value = 42")
    assert len(calls) == 1
