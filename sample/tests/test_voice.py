import httpx
import pytest
import asyncio
from types import SimpleNamespace
from pydantic import ValidationError
from contextlib import asynccontextmanager
from pydantic_ai import Agent as RealtimeAgent
from sample.src.shell import RemoteShell
from pydantic_ai.messages import TextPart, ModelResponse
from sample.src.workspace import Workspace
from sample.src.agents.voice import VoiceCommand, VoiceSession
from sample.src.agents.script import Agent
from pydantic_ai.models.function import FunctionModel


@pytest.fixture
def call_factory(monkeypatch):
    @asynccontextmanager
    async def create(*, phase=None, outcome="complete"):
        entered = asyncio.Event()
        release = asyncio.Event()
        working = asyncio.Event()
        finish_work = asyncio.Event()
        events = []
        offers = []
        hangups = []
        queued = []
        tools = {}

        async def checkpoint(name):
            if phase == name:
                entered.set()
                await release.wait()

        async def request(request):
            if request.url.path == "/libraries":
                await checkpoint("inventory")
                return httpx.Response(200, json=[])
            if request.url.path == "/execute/stream":
                working.set()
                await finish_work.wait()
                return httpx.Response(
                    200,
                    text='{"type":"result","stdout":"accepted","stderr":"","exception":null,"done":true}\n{"type":"execution_finished"}\n',
                )
            assert request.url.path == "/ui/context"
            return httpx.Response(200, json={"charts": []})

        class Session:
            def enqueue(self, text):
                queued.append(text)

            async def __aiter__(self):
                await checkpoint("active")
                if outcome == "error":
                    raise RuntimeError("provider failed")
                return
                yield

        class Realtime:
            async def answer_webrtc_offer(self, sdp):
                offers.append(sdp)
                await checkpoint("setup")
                if outcome == "setup_error":
                    raise RuntimeError("offer failed")
                return SimpleNamespace(sdp="answer", session="provider-call")

            @asynccontextmanager
            async def session(self, *, provider_session):
                assert provider_session == "provider-call"
                await checkpoint("sideband")
                yield Session()

            async def hang_up(self, session):
                hangups.append(session)
                if outcome == "hangup_error":
                    raise RuntimeError("hangup failed")

        def realtime(agent, *args, **kwargs):
            tools.update(agent._function_toolset.tools)
            if outcome == "factory_error":
                raise RuntimeError("realtime setup failed")
            return Realtime()

        monkeypatch.setattr(RealtimeAgent, "realtime", realtime)
        model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("done()")], finish_reason="stop"))
        async with httpx.AsyncClient(base_url="http://execution", transport=httpx.MockTransport(request)) as client:
            workspace = Workspace(Agent(model, shell=RemoteShell(client)))
            voice = VoiceSession(None, workspace, events.append)
            try:
                yield SimpleNamespace(
                    voice=voice,
                    workspace=workspace,
                    events=events,
                    entered=entered,
                    release=release,
                    offers=offers,
                    hangups=hangups,
                    queued=queued,
                    tools=tools,
                    working=working,
                    finish_work=finish_work,
                )
            finally:
                finish_work.set()
                await voice.stop(notify=False)
                await workspace.close()

    return create


@pytest.mark.parametrize("offer", [{}, {"sdp": None}, {"sdp": ""}, {"sdp": "x" * 65537}])
def test_invalid_start_never_reserves_call(call_factory, offer):
    async def scenario():
        async with call_factory() as call:
            with pytest.raises(ValidationError) as error:
                await call.voice.handle(VoiceCommand.model_validate({"type": "voice", "action": "start", **offer}))
            if not offer or offer["sdp"] is None:
                assert "Voice start requires an SDP offer." in str(error.value)
            assert call.voice.task is None
            assert call.events == []
            assert call.offers == []
            assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())


@pytest.mark.parametrize("sdp", ["offer", "x" * 65536])
def test_valid_start_and_stop_without_offer(call_factory, sdp):
    async def scenario():
        async with call_factory(phase="active") as call:
            await call.voice.handle(VoiceCommand(type="voice", action="start", sdp=sdp))
            await asyncio.wait_for(call.entered.wait(), 1)
            with pytest.raises(RuntimeError, match="already active"):
                await call.voice.handle(VoiceCommand(type="voice", action="start", sdp="duplicate"))
            await call.voice.handle(VoiceCommand(type="voice", action="stop"))
            await call.voice.handle(VoiceCommand(type="voice", action="stop"))
            assert call.offers == [sdp]
            assert [event["name"] for event in call.events] == ["answer", "closed"]
            assert call.hangups == ["provider-call"]
            assert call.voice.task is None
            assert not call.workspace.results

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", [None, "inventory", "setup", "sideband", "active"])
@pytest.mark.parametrize("notify", [False, True])
def test_stop_notifies_once_in_every_phase(call_factory, phase, notify):
    async def scenario():
        async with call_factory(phase=phase) as call:
            await call.voice.handle(VoiceCommand(type="voice", action="start", sdp="offer"))
            task = call.voice.task
            if phase is not None:
                await asyncio.wait_for(call.entered.wait(), 1)
            await call.voice.stop(notify=notify)
            await call.voice.stop()
            await call.voice.stop(notify=False)
            assert task.cancelled()
            assert [event["name"] for event in call.events].count("closed") == int(notify)
            assert not any(event["name"] == "error" for event in call.events)
            assert call.hangups == (["provider-call"] if phase in {"sideband", "active"} else [])
            assert call.voice.task is None
            assert not call.workspace.results
            assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["complete", "error", "setup_error", "factory_error", "hangup_error"])
def test_completion_and_errors_notify_once(call_factory, outcome):
    async def scenario():
        async with call_factory(outcome=outcome) as call:
            await call.voice.handle(VoiceCommand(type="voice", action="start", sdp="offer"))
            task = call.voice.task
            await task
            assert task.done()
            await call.voice.stop()
            await call.voice.stop()
            names = [event["name"] for event in call.events]
            assert names.count("closed") == 1
            assert names[-1] == "closed"
            assert names.count("error") == int(outcome in {"error", "setup_error", "factory_error"})
            assert call.hangups == ([] if outcome in {"setup_error", "factory_error"} else ["provider-call"])
            assert not call.workspace.results

    asyncio.run(scenario())


def test_direct_task_cancellation_notifies_once(call_factory):
    async def scenario():
        async with call_factory(phase="inventory") as call:
            await call.voice.handle(VoiceCommand(type="voice", action="start", sdp="offer"))
            task = call.voice.task
            await asyncio.wait_for(call.entered.wait(), 1)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await call.voice.stop()
            assert [event["name"] for event in call.events] == ["closed"]

    asyncio.run(scenario())


def test_hangup_preserves_accepted_workspace_work(call_factory):
    async def scenario():
        async with call_factory(phase="active") as call:
            await call.voice.handle(VoiceCommand(type="voice", action="start", sdp="offer"))
            await asyncio.wait_for(call.entered.wait(), 1)
            assert await call.tools["worker"].function("print('accepted')") == "Task accepted; completion will arrive separately."
            task = call.workspace.task
            await asyncio.wait_for(call.working.wait(), 1)
            assert call.workspace.busy
            await call.voice.stop()
            assert not task.cancelled()
            assert not task.done()
            call.finish_work.set()
            await call.workspace.close()
            assert call.workspace.recent_tasks[0]["outcome"] == "completed"
            assert not call.workspace.results
            assert [event["name"] for event in call.events] == ["answer", "closed"]

    asyncio.run(scenario())
