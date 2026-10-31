import pytest
import asyncio
from api.websocket import run_websocket
from starlette.websockets import WebSocketDisconnect


@pytest.mark.parametrize("first", [0, 1])
@pytest.mark.parametrize("error", [None, WebSocketDisconnect(), RuntimeError("socket failed")])
def test_first_completion_drains_peer(first, error):
    async def scenario():
        started = asyncio.Event()
        drained = asyncio.Event()

        async def finish():
            await started.wait()
            if error is not None:
                raise error

        async def peer():
            started.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                drained.set()

        directions = [finish(), peer()]
        if first:
            directions.reverse()
        if isinstance(error, RuntimeError):
            with pytest.raises(RuntimeError, match="socket failed") as caught:
                await run_websocket(*directions)
            assert caught.value is error
        else:
            await run_websocket(*directions)
        assert drained.is_set()
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())


def test_cancellation_cancels_both_before_draining_either():
    async def scenario():
        started = [asyncio.Event(), asyncio.Event()]
        cancelling = [asyncio.Event(), asyncio.Event()]
        drained = []

        async def direction(index):
            started[index].set()
            try:
                await asyncio.Future()
            finally:
                cancelling[index].set()
                await cancelling[1 - index].wait()
                drained.append(index)

        runner = asyncio.create_task(run_websocket(direction(0), direction(1)))
        await asyncio.gather(*(event.wait() for event in started))
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(runner, timeout=1)
        assert sorted(drained) == [0, 1]
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())


def test_first_failure_survives_peer_cleanup_failure():
    async def scenario():
        started = asyncio.Event()
        failure = RuntimeError("first failure")
        drained = asyncio.Event()

        async def fail():
            await started.wait()
            raise failure

        async def peer():
            started.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                drained.set()
                raise ValueError("cleanup failure")

        with pytest.raises(RuntimeError, match="first failure") as caught:
            await run_websocket(fail(), peer())
        assert caught.value is failure
        assert drained.is_set()
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("disconnect", "cleanup_error"),
    [(False, ValueError("cleanup failure")), (True, ValueError("cleanup failure")), (False, WebSocketDisconnect())],
    ids=["normal-finish-cleanup-fails", "disconnect-cleanup-fails", "cleanup-disconnects"],
)
def test_peer_cleanup_outcomes(disconnect, cleanup_error):
    async def scenario():
        started = asyncio.Event()

        async def finish():
            await started.wait()
            if disconnect:
                raise WebSocketDisconnect()

        async def peer():
            started.set()
            try:
                await asyncio.Future()
            finally:
                raise cleanup_error

        if isinstance(cleanup_error, ValueError):
            with pytest.raises(ValueError, match="cleanup failure"):
                await run_websocket(finish(), peer())
        else:
            await run_websocket(finish(), peer())
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())


def test_simultaneous_disconnect_does_not_hide_failure():
    async def scenario():
        async def disconnect():
            raise WebSocketDisconnect()

        async def fail():
            raise ValueError("socket failed")

        with pytest.raises(ValueError, match="socket failed"):
            await run_websocket(disconnect(), fail())
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())
