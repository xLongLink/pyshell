import os
import sys
import pytest
import asyncio
import subprocess
from types import SimpleNamespace
from pathlib import Path
from pydantic import ValidationError
from unittest.mock import Mock, AsyncMock


def test_application_import_does_not_validate_provider_credentials():
    subprocess.run(
        [sys.executable, "-c", "import sample.main"],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "OPENAI_API_KEY": ""},
        check=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize("api_key", [None, ""])
def test_startup_rejects_invalid_credentials_before_creating_resources(monkeypatch, api_key):
    from sample import main

    monkeypatch.setitem(main.Settings.model_config, "env_file", None)
    if api_key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", api_key)
    provider = Mock()
    monkeypatch.setattr(main, "OpenAIProvider", provider)

    async def scenario():
        with pytest.raises(ValidationError):
            async with main.lifespan(main.app):
                pytest.fail("Invalid credentials allowed startup")

    asyncio.run(scenario())
    provider.assert_not_called()


@pytest.mark.parametrize(
    "origin",
    [None, "null", "https://evil.example", "http://localhost:8011", "http://localhost:8010.evil.example", "https://localhost:8010"],
)
def test_socket_rejects_untrusted_origins_before_accepting(origin):
    from sample import main

    socket = SimpleNamespace(headers={} if origin is None else {"origin": origin}, accept=AsyncMock(), close=AsyncMock())

    asyncio.run(main.workspace_socket(socket))

    socket.accept.assert_not_awaited()
    socket.close.assert_awaited_once_with(code=1008)


def test_overloaded_socket_stops_voice_without_cancelling_accepted_work(monkeypatch):
    from sample import main
    from sample.src.workspace import Workspace

    async def scenario():
        sending = asyncio.Event()
        working = asyncio.Event()
        release = asyncio.Event()
        stopped = []
        close_codes = []

        class Agent:
            async def arun(self, instruction, on_phase):
                working.set()
                await release.wait()
                return "Completed"

        workspace = Workspace(Agent())
        workspace.agent.shell = SimpleNamespace(uncertain=False)

        class Voice:
            def __init__(self, *args):
                pass

            async def stop(self, *, notify):
                stopped.append(notify)

        class Socket:
            headers = {"origin": "http://localhost:8010"}
            app = SimpleNamespace(state=SimpleNamespace(workspace=workspace, realtime_model=None))

            async def accept(self):
                pass

            async def send_json(self, event):
                sending.set()
                await asyncio.Future()

            async def receive_text(self):
                await sending.wait()
                workspace.start("Keep working")
                for _ in range(255):
                    workspace.publish({"type": "error", "text": "Browser is too slow"})
                assert workspace.listeners
                workspace.publish({"type": "error", "text": "Browser is too slow"})
                await asyncio.Future()

            async def close(self, *, code):
                close_codes.append(code)

        monkeypatch.setattr(main, "VoiceSession", Voice)
        await asyncio.wait_for(main.workspace_socket(Socket()), timeout=1)
        await asyncio.wait_for(working.wait(), timeout=1)
        assert close_codes == [1013]
        assert stopped == [False]
        assert not workspace.listeners
        assert workspace.busy
        assert not workspace.task.done()
        release.set()
        await workspace.close()
        assert workspace.recent_tasks[0]["outcome"] == "completed"
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())
