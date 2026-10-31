import pytest
import asyncio
from api.websocket import run_websocket
from starlette.websockets import WebSocketDisconnect


@pytest.mark.parametrize(
    ("count", "first", "error"),
    [(2, 0, None), (2, 1, None), (3, 2, None), (3, 2, WebSocketDisconnect()), (3, 2, RuntimeError("socket failed"))],
)
def test_first_completion_drains_peer(count, first, error):
    async def scenario():
        started = [asyncio.Event() for _ in range(count - 1)]
        drained = []

        async def finish():
            await asyncio.gather(*(event.wait() for event in started))
            if error is not None:
                raise error

        async def peer(index):
            started[index].set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                drained.append(index)

        directions = [peer(index) for index in range(count - 1)]
        directions.insert(first, finish())
        if isinstance(error, RuntimeError):
            with pytest.raises(RuntimeError, match="socket failed") as caught:
                await run_websocket(*directions)
            assert caught.value is error
        else:
            await run_websocket(*directions)
        assert sorted(drained) == list(range(count - 1))
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())


def test_cancellation_cancels_all_before_draining_any():
    async def scenario():
        count = 3
        started = [asyncio.Event() for _ in range(count)]
        cancelling = [asyncio.Event() for _ in range(count)]
        drained = []

        async def direction(index):
            started[index].set()
            try:
                await asyncio.Future()
            finally:
                cancelling[index].set()
                await asyncio.gather(*(event.wait() for event in cancelling))
                drained.append(index)

        runner = asyncio.create_task(run_websocket(*(direction(index) for index in range(count))))
        await asyncio.gather(*(event.wait() for event in started))
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(runner, timeout=1)
        assert sorted(drained) == list(range(count))
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("primary_error", "cleanup_error"),
    [
        (None, ValueError("cleanup failure")),
        (WebSocketDisconnect(), ValueError("cleanup failure")),
        (None, WebSocketDisconnect()),
        (RuntimeError("first failure"), ValueError("cleanup failure")),
    ],
    ids=["normal-finish-cleanup-fails", "disconnect-cleanup-fails", "cleanup-disconnects", "primary-failure-wins"],
)
def test_peer_cleanup_outcomes(primary_error, cleanup_error):
    async def scenario():
        started = asyncio.Event()
        drained = []

        async def finish():
            await started.wait()
            if primary_error is not None:
                raise primary_error

        async def peer():
            started.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                drained.append(True)
                raise cleanup_error

        failure = primary_error if isinstance(primary_error, RuntimeError) else cleanup_error
        if isinstance(failure, WebSocketDisconnect):
            await run_websocket(finish(), peer(), peer())
        else:
            with pytest.raises(type(failure)) as caught:
                await run_websocket(finish(), peer(), peer())
            assert caught.value is failure
        assert len(drained) == 2
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())


def test_simultaneous_disconnect_does_not_hide_failure():
    async def scenario():
        async def disconnect():
            raise WebSocketDisconnect()

        async def fail():
            raise ValueError("socket failed")

        async def wait():
            await asyncio.Future()

        with pytest.raises(ValueError, match="socket failed"):
            await run_websocket(disconnect(), fail(), wait())
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())
