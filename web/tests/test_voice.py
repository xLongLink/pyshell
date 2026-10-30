import json
import asyncio
from types import SimpleNamespace
from contextlib import asynccontextmanager
from unittest.mock import Mock, AsyncMock
from src.ui.session import UI
from web.src.agents import voice
from web.src.workspace import Command, Workspace


class Realtime:
    def __init__(self):
        self.entered = asyncio.Event()
        self.messages = []
        self.delivered = asyncio.Event()
        self.hang_up = AsyncMock()

    async def answer_webrtc_offer(self, sdp):
        return SimpleNamespace(sdp="answer", session="provider")

    @asynccontextmanager
    async def session(self, **kwargs):
        self.entered.set()
        yield self

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Event().wait()

    def enqueue(self, message):
        self.messages.append(message)
        if "Workspace task result" in message:
            self.delivered.set()


def install_agent(monkeypatch, realtime, tools):
    settings = {}

    class Agent:
        def __init__(self, **kwargs):
            settings.update(kwargs)

        def tool_plain(self, function):
            tools[function.__name__] = function
            return function

        def realtime(self, *args, **kwargs):
            return realtime

    monkeypatch.setattr(voice, "Agent", Agent)
    return settings


def test_voice_delegates_to_shared_workspace_and_hangup_does_not_cancel_work(monkeypatch):
    async def run():
        release = asyncio.Event()
        started = asyncio.Event()
        realtime = Realtime()
        tools = {}

        install_agent(monkeypatch, realtime, tools)

        async def execute(instruction, emit):
            started.set()
            await release.wait()
            return "Confirmed result: 42"

        shell = SimpleNamespace(uncertain=False)
        agent = SimpleNamespace(shell=shell, pending_feedback=None, arun=execute)
        workspace = Workspace(agent)
        events = []

        async def emit(event):
            events.append(event)

        call = voice.VoiceSession(None, workspace, emit)
        await call.handle(voice.VoiceCommand(type="voice", action="start", sdp="offer"))
        await realtime.entered.wait()
        await asyncio.sleep(0)
        assert events[0] == {"type": "voice", "name": "answer", "sdp": "answer"}
        assert "submitted" in await tools["worker"]("Calculate 42")
        await started.wait()
        assert "not submitted" in await tools["worker"]("Do not overlap")

        # Hangup releases voice resources without cancelling accepted work.
        await call.stop()
        assert not workspace.task.cancelled()
        assert workspace.busy
        assert events[-1] == {"type": "voice", "name": "closed"}
        realtime.hang_up.assert_awaited_once_with("provider")
        release.set()
        await workspace.task
        assert workspace.status()["latest_outcome"]["result"] == "Confirmed result: 42"
        assert not any("Confirmed result" in message for message in realtime.messages)

        # A new call reuses the workspace and receives new task results.
        await call.handle(voice.VoiceCommand(type="voice", action="start", sdp="offer"))
        await asyncio.sleep(0)
        assert "submitted" in await tools["worker"]("Continue with the same workspace")
        await asyncio.wait_for(realtime.delivered.wait(), 1)
        assert any("Confirmed result: 42" in message for message in realtime.messages)
        await call.stop()

    asyncio.run(run())


def test_reconnection_after_failure_keeps_old_tools_disconnected(monkeypatch):
    class FailedRealtime(Realtime):
        async def __anext__(self):
            raise RuntimeError("Sideband failed")

    async def run():
        async def execute(instruction, emit):
            return "Confirmed after reconnect"

        workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(uncertain=False), arun=execute))
        failed = FailedRealtime()
        old_tools = {}
        install_agent(monkeypatch, failed, old_tools)
        events = []

        async def emit(event):
            events.append(event)

        call = voice.VoiceSession(None, workspace, emit)
        await call.handle(voice.VoiceCommand(type="voice", action="start", sdp="offer"))
        await call.task
        assert any(event.get("name") == "error" for event in events)
        assert events[-1]["name"] == "closed"
        assert not workspace.results
        failed.hang_up.assert_awaited_once_with("provider")

        realtime = Realtime()
        tools = {}
        install_agent(monkeypatch, realtime, tools)
        await call.handle(voice.VoiceCommand(type="voice", action="start", sdp="new offer"))
        try:
            await asyncio.wait_for(realtime.entered.wait(), 1)
            assert "not connected" in await old_tools["worker"]("Stale task")
            assert workspace.task is None
            assert "submitted" in await tools["worker"]("New task")
            await workspace.task
            assert workspace.status()["latest_outcome"]["result"] == "Confirmed after reconnect"
            await asyncio.wait_for(realtime.delivered.wait(), 1)
            assert any("Confirmed after reconnect" in message for message in realtime.messages)
            assert not any("Confirmed after reconnect" in message for message in failed.messages)
        finally:
            await call.stop()
        assert not workspace.results

    asyncio.run(run())


def test_voice_inspects_previous_work_and_hangs_up_without_cancelling_python(monkeypatch):
    async def run():
        realtime = Realtime()
        tools = {}
        release = asyncio.Event()

        settings = install_agent(monkeypatch, realtime, tools)
        ui = UI()

        async def execute(instruction, emit):
            if instruction == "slow":
                await release.wait()
            else:
                ui.on_publish = emit
                ui.plot([{"type": "bar", "x": ["A"], "y": [42]}], surface_id="sales")
            return "Confirmed answer: 42"

        workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(uncertain=False), arun=execute))
        await workspace.start(Command(type="run", instruction="Calculate answer"))
        events = []

        async def emit(event):
            events.append(event)

        call = voice.VoiceSession(None, workspace, emit)
        await call.handle(voice.VoiceCommand(type="voice", action="start", sdp="offer"))
        await asyncio.sleep(0)
        assert "Calculate answer" in realtime.messages[0]
        assert "persistent Python workspace" in settings["instructions"]
        assert "Confirmed answer: 42" in realtime.messages[0]
        context = json.loads(await tools["inspect_workspace"]())
        assert context["recent_tasks"][0]["instruction"] == "Calculate answer"
        assert context["charts"][0]["surface_id"] == "sales"
        assert context["status"]["execution_uncertain"] is False
        task = workspace.start(Command(type="run", instruction="slow"))
        status = json.loads(await tools["inspect_workspace"]())["status"]
        assert status["busy"] is True
        assert status["instruction"] == "slow"
        result = await tools["worker"]("Voice task")
        assert "not submitted" in result
        assert "busy" in result
        assert workspace.task is task
        handle = Mock(spec=asyncio.TimerHandle)
        call_later = Mock(return_value=handle)
        with monkeypatch.context() as patch:
            patch.setattr(asyncio.get_running_loop(), "call_later", call_later)
            assert "end shortly" in await tools["end_call"]()
            await tools["end_call"]()
        call_later.assert_called_once()
        delay, callback = call_later.call_args.args
        assert delay == 2
        await asyncio.wait_for(callback(), 1)
        handle.cancel.assert_called_once_with()
        assert call.task is None
        assert events[-1]["name"] == "closed"
        realtime.hang_up.assert_awaited_once()
        assert not task.cancelled()
        assert workspace.busy
        release.set()
        await task

    asyncio.run(run())
