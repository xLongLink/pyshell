import asyncio
from collections.abc import Coroutine
from starlette.websockets import WebSocketDisconnect


async def run_websocket(
    send: Coroutine[None, None, object], receive: Coroutine[None, None, object], *additional: Coroutine[None, None, object]
) -> None:
    """Run socket operations until one ends, then drain all tasks."""
    tasks = tuple(asyncio.create_task(operation) for operation in (send, receive, *additional))
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass
    finally:
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, (asyncio.CancelledError, WebSocketDisconnect)):
            raise result
