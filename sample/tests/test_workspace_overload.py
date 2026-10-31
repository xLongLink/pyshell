import pytest
import asyncio
import importlib
from types import SimpleNamespace
from sample.src.workspace import Workspace, Subscription


@pytest.mark.parametrize("event", [{"type": "instruction", "text": "calculate"}, {"type": "error", "text": "failed"}])
def test_overloaded_subscriber_does_not_block_broadcast_or_work(event):
    async def run():
        async def arun(instruction, publish):
            if event["type"] == "error":
                raise RuntimeError("failed")
            publish(event)
            return "completed"

        workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(uncertain=False), arun=arun))
        slow = Subscription(workspace, maxsize=1)
        healthy = Subscription(workspace)
        healthy.queue.get_nowait()
        await workspace.start("calculate")
        assert slow.closed.is_set()
        assert slow not in workspace.listeners
        assert slow.queue.get_nowait() == {"type": "snapshot"}
        assert healthy.queue.get_nowait() == {"type": "instruction", "text": "calculate"}
        assert healthy.queue.get_nowait() == event
        assert not healthy.closed.is_set()
        assert workspace.recent_tasks[-1]["outcome"] == ("failed" if event["type"] == "error" else "completed")
        assert not workspace.busy

    asyncio.run(run())


@pytest.mark.parametrize("name", ["answer", "error", "closed"])
def test_voice_controls_use_the_same_overflow_policy(name):
    async def run():
        workspace = Workspace(SimpleNamespace())
        subscription = Subscription(workspace, maxsize=1)
        await subscription.emit({"type": "voice", "name": name})
        assert subscription.closed.is_set()
        assert not workspace.listeners
        await subscription.emit({"type": "voice", "name": "closed"})
        assert subscription.queue.qsize() == 1

    asyncio.run(run())


@pytest.mark.parametrize("producer", ["command", "workspace", "voice"])
@pytest.mark.parametrize("blocked_close", [False, True])
def test_socket_overflow_closes_transport_and_voice_without_cancelling_work(monkeypatch, producer, blocked_close):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    main = importlib.import_module("sample.main")

    async def run():
        sending = asyncio.Event()
        release = asyncio.Event()
        started = asyncio.Event()
        stopped = asyncio.Event()
        subscriptions = []
        closes = []
        commands = asyncio.Queue()

        async def arun(instruction, publish):
            started.set()
            await release.wait()
            return "completed"

        workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(uncertain=False), arun=arun))
        healthy = Subscription(workspace)
        healthy.queue.get_nowait()

        def subscribe(workspace):
            subscription = Subscription(workspace, maxsize=1)
            subscriptions.append(subscription)
            return subscription

        class Voice:
            def __init__(self, model, workspace, emit):
                self.emit = emit

            async def handle(self, command):
                await self.emit({"type": "voice", "name": "answer", "sdp": "answer"})
                await self.emit({"type": "voice", "name": "closed"})

            async def stop(self, *, notify):
                assert not notify
                stopped.set()

        class Socket:
            headers = {"origin": "http://localhost:8010"}
            app = SimpleNamespace(state=SimpleNamespace(workspace=workspace, realtime_model=object()))

            async def accept(self):
                pass

            async def send_json(self, event):
                sending.set()
                await asyncio.Event().wait()

            async def receive_text(self):
                return await commands.get()

            async def close(self, *, code):
                closes.append(code)
                if blocked_close:
                    await asyncio.Event().wait()

        monkeypatch.setattr(main, "Subscription", subscribe)
        monkeypatch.setattr(main, "VoiceSession", Voice)
        timeout = asyncio.timeout
        monkeypatch.setattr(main.asyncio, "timeout", lambda delay: timeout(0 if delay == 5 else delay))
        socket_task = asyncio.create_task(main.workspace_socket(Socket()))
        try:
            async with asyncio.timeout(2):
                await sending.wait()
                work = workspace.start("calculate")
                await started.wait()
                if producer == "command":
                    commands.put_nowait('{"type":"invalid"}')
                elif producer == "voice":
                    commands.put_nowait('{"type":"voice","action":"start","sdp":"offer"}')
                else:
                    workspace.publish({"type": "error", "text": "failed"})
                if blocked_close:
                    with pytest.raises(TimeoutError):
                        await socket_task
                else:
                    await socket_task
                assert closes == [1013]
                assert stopped.is_set()
                assert subscriptions[0].closed.is_set()
                assert workspace.listeners == {healthy}
                assert not work.done()
                release.set()
                await work
                assert workspace.recent_tasks[-1]["outcome"] == "completed"
                assert healthy.queue.get_nowait() == {"type": "instruction", "text": "calculate"}
        finally:
            release.set()
            socket_task.cancel()
            await asyncio.gather(socket_task, return_exceptions=True)
            await workspace.close()

    asyncio.run(run())
