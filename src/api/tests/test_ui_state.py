import gc
import asyncio
from weakref import ref
from api.ui.state import UIState
from src.api.tests.ui_helpers import content, creation


def test_unsubscribe_before_delivery_does_not_deliver_to_detached_queue():
    async def run():
        state = UIState()
        detached = state.subscribe()
        detached.get_nowait()
        state.publish([creation("sales"), content("sales")])
        state.unsubscribe(detached)
        replacement = state.subscribe()
        state.publish([content("sales", "Latest")])
        await asyncio.sleep(0)
        assert detached.empty()
        assert not state.pending
        assert replacement.qsize() == 1
        assert replacement.get_nowait() == state.snapshot()
        state.unsubscribe(replacement)

    asyncio.run(run())


def test_each_event_loop_receives_the_latest_snapshot_independently():
    state = UIState()
    loops = [asyncio.new_event_loop(), asyncio.new_event_loop()]

    async def subscribe():
        return state.subscribe()

    try:
        queues = [loop.run_until_complete(subscribe()) for loop in loops]
        state.publish([creation("sales"), content("sales")])
        state.publish([content("sales", "Latest")])
        loops[0].run_until_complete(asyncio.sleep(0))
        assert queues[0].qsize() == 1
        assert queues[0].get_nowait() == state.snapshot()
        assert queues[1].get_nowait()["events"] == []
        loops[1].run_until_complete(asyncio.sleep(0))
        assert queues[1].qsize() == 1
        assert queues[1].get_nowait() == state.snapshot()
        for queue in queues:
            state.unsubscribe(queue)
    finally:
        for loop in loops:
            loop.close()


def test_pending_delivery_does_not_retain_a_closed_unsubscribed_loop():
    state = UIState()
    loop = asyncio.new_event_loop()

    async def subscribe():
        return state.subscribe()

    queue = loop.run_until_complete(subscribe())
    state.publish([creation("sales"), content("sales")])
    state.unsubscribe(queue)
    closed_loop = ref(loop)
    loop.close()
    del loop
    gc.collect()
    assert closed_loop() is None
    assert not state.pending


def test_public_snapshot_containers_do_not_mutate_shared_delivery():
    async def run():
        state = UIState()
        first = state.subscribe()
        second = state.subscribe()
        state.publish([creation("sales"), content("sales")])
        await asyncio.sleep(0)
        shared = first.get_nowait()
        assert second.get_nowait() == shared
        public = state.snapshot()
        public["events"][0]["messages"].clear()
        public["events"].clear()
        public["placements"]["sales"]["pinned"] = True
        assert state.snapshot() == shared
        assert len(shared["events"][0]["messages"]) == 2
        state.unsubscribe(first)
        state.unsubscribe(second)

    asyncio.run(run())
