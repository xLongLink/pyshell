import asyncio
from types import SimpleNamespace
from web.src import voice
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
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


def test_voice_delegates_to_shared_workspace_and_hangup_does_not_cancel_work(monkeypatch):
    async def run():
        release = asyncio.Event()
        started = asyncio.Event()
        realtime = Realtime()
        tools = {}

        class Agent:
            def __init__(self, **kwargs):
                pass

            def tool_plain(self, function):
                tools[function.__name__] = function
                return function

            def realtime(self, *args, **kwargs):
                return realtime

        monkeypatch.setattr(voice, "Agent", Agent)

        async def execute(instruction, emit):
            started.set()
            await release.wait()
            return "Confirmed result: 42"

        shell = SimpleNamespace(uncertain=False)
        agent = SimpleNamespace(model="test", shell=shell, pending_feedback=None, arun=execute)
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
        await call.stop()
        assert not workspace.task.cancelled()
        assert workspace.busy
        assert events[-1] == {"type": "voice", "name": "closed"}
        realtime.hang_up.assert_awaited_once_with("provider")
        release.set()
        assert await workspace.task == "Confirmed result: 42"
        assert not any("Confirmed result" in message for message in realtime.messages)

        await call.handle(voice.VoiceCommand(type="voice", action="start", sdp="offer"))
        await asyncio.sleep(0)
        assert "submitted" in await tools["worker"]("Continue with the same workspace")
        await asyncio.wait_for(realtime.delivered.wait(), 1)
        assert any("Confirmed result: 42" in message for message in realtime.messages)
        await call.stop()

    asyncio.run(run())


def test_voice_rejects_a_task_while_text_work_is_running(monkeypatch):
    async def run():
        release = asyncio.Event()
        tools = {}
        realtime = Realtime()

        class Agent:
            def __init__(self, **kwargs):
                pass

            def tool_plain(self, function):
                tools[function.__name__] = function
                return function

            def realtime(self, *args, **kwargs):
                return realtime

        monkeypatch.setattr(voice, "Agent", Agent)

        async def execute(instruction, emit):
            await release.wait()
            return "Text completed"

        workspace = Workspace(SimpleNamespace(model="test", shell=SimpleNamespace(uncertain=False), pending_feedback=None, arun=execute))
        text = workspace.start(Command(type="run", instruction="Text task"))
        call = voice.VoiceSession(None, workspace, AsyncMock())
        await call.handle(voice.VoiceCommand(type="voice", action="start", sdp="offer"))
        await realtime.entered.wait()
        await asyncio.sleep(0)
        result = await tools["worker"]("Voice task")
        assert "not submitted" in result
        assert "busy" in result
        assert workspace.task is text
        await call.stop()
        release.set()
        await text

    asyncio.run(run())
