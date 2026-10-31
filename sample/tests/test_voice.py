import httpx2
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
    async def create(*, phase=None, outcome="complete", hold_cleanup=False):
        entered = asyncio.Event()
        release = asyncio.Event()
        working = asyncio.Event()
        finish_work = asyncio.Event()
        cleanup_started = asyncio.Event()
        cleanup_release = asyncio.Event()
        completed = asyncio.Event()
        events = []
        offers = []
        hangups = []
        tools = {}

        async def checkpoint(name):
            if phase == name:
                entered.set()
                await release.wait()

        async def request(request):
            if request.url.path == "/libraries":
                await checkpoint("inventory")
                return httpx2.Response(200, json=[])
            if request.url.path == "/execute/stream":
                working.set()
                await finish_work.wait()
                return httpx2.Response(
                    200,
                    text='{"type":"result","stdout":"accepted","stderr":"","exception":null,"done":true}\n{"type":"execution_finished"}\n',
                )
            assert request.url.path == "/ui/context"
            return httpx2.Response(200, json={"charts": []})

        class Session:
            def enqueue(self, text):
                pass

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
                cleanup_started.set()
                if hold_cleanup:
                    await cleanup_release.wait()
                if outcome == "hangup_error":
                    raise RuntimeError("hangup failed")
                completed.set()

        def realtime(agent, *args, **kwargs):
            tools.update(agent._function_toolset.tools)
            if outcome == "factory_error":
                raise RuntimeError("realtime setup failed")
            return Realtime()

        monkeypatch.setattr(RealtimeAgent, "realtime", realtime)
        model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("done()")], finish_reason="stop"))
        async with httpx2.AsyncClient(base_url="http://execution", transport=httpx2.MockTransport(request)) as client:
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
                    tools=tools,
                    working=working,
                    finish_work=finish_work,
                    cleanup_started=cleanup_started,
                    cleanup_release=cleanup_release,
                    completed=completed,
                )
            finally:
                finish_work.set()
                cleanup_release.set()
                await voice.stop(notify=False)
                await workspace.close()

    return create


@pytest.mark.parametrize("offer", [{}, {"sdp": None}, {"sdp": ""}, {"sdp": "x" * 65537}])
def test_start_command_rejects_invalid_offers(offer):
    with pytest.raises(ValidationError) as error:
        VoiceCommand.model_validate({"type": "voice", "action": "start", **offer})
    if not offer or offer["sdp"] is None:
        assert "Voice start requires an SDP offer." in str(error.value)


def test_valid_start_and_stop_without_offer(call_factory):
    assert VoiceCommand(type="voice", action="start", sdp="offer").sdp == "offer"
    sdp = "x" * 65536

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


@pytest.mark.parametrize("stopped", [False, True])
def test_old_completion_cannot_close_restarted_call(call_factory, stopped):
    async def scenario():
        async with call_factory(phase="active") as call:
            start = VoiceCommand(type="voice", action="start", sdp="offer")
            await call.voice.handle(start)
            old_task = call.voice.task
            await asyncio.wait_for(call.entered.wait(), 1)

            async def restart():
                await call.completed.wait()
                assert old_task.done()
                call.release.clear()
                await call.voice.handle(start)
                new_task = call.voice.task
                assert new_task is not old_task
                await asyncio.sleep(0)
                assert call.voice.task is new_task
                assert not any(event["name"] == "closed" for event in call.events)
                await call.voice.stop()

            restarting = asyncio.create_task(restart())
            await asyncio.sleep(0)
            if stopped:
                await call.voice.stop()
            else:
                call.release.set()
            await asyncio.wait_for(restarting, 1)
            assert [event["name"] for event in call.events].count("closed") == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("silent", [False, True])
def test_concurrent_stop_keeps_guard_through_cleanup(call_factory, silent):
    async def scenario():
        async with call_factory(phase="active", hold_cleanup=True) as call:
            start = VoiceCommand(type="voice", action="start", sdp="offer")
            await call.voice.handle(start)
            task = call.voice.task
            await asyncio.wait_for(call.entered.wait(), 1)
            first = asyncio.create_task(call.voice.stop())
            await asyncio.wait_for(call.cleanup_started.wait(), 1)
            second = asyncio.create_task(call.voice.stop(notify=not silent))
            await asyncio.sleep(0)
            assert not task.done()
            assert call.voice.task is task
            assert call.hangups == ["provider-call"]
            with pytest.raises(RuntimeError, match="already active"):
                await call.voice.handle(start)
            call.cleanup_release.set()
            await asyncio.gather(first, second)
            await call.voice.stop()
            assert [event["name"] for event in call.events].count("closed") == int(not silent)
            assert call.voice.task is None

    asyncio.run(scenario())
