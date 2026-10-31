import asyncio
from types import SimpleNamespace


def test_overloaded_socket_stops_voice_without_cancelling_accepted_work(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from sample import main
    from sample.src.workspace import Workspace, Subscription

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
                workspace.publish({"type": "error", "text": "Browser is too slow"})
                await asyncio.Future()

            async def close(self, *, code):
                close_codes.append(code)

        monkeypatch.setattr(main, "VoiceSession", Voice)
        monkeypatch.setattr(main, "Subscription", lambda owner: Subscription(owner, maxsize=1))
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
