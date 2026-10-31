import gc
import pytest
import asyncio
from weakref import ref
from api.ui.state import UIState
from api.ui.session import UI
from src.api.tests.ui_helpers import content, creation


@pytest.mark.parametrize(
    "messages",
    [
        [creation("new")],
        [creation("new"), content("sales")],
        [creation("new"), {"updateComponents": {"surfaceId": "new"}}],
        [{"deleteSurface": {"surfaceId": "sales"}}, creation("sales")],
        [content("sales", "Changed"), creation("new")],
        [creation("new"), content("new"), creation("incomplete")],
    ],
)
def test_incomplete_creation_rejects_entire_batch_and_preserves_reconnect(messages):
    async def run():
        state = UIState()
        for index in range(5):
            identifier = "sales" if index == 0 else f"chart_{index}"
            state.publish([creation(identifier), content(identifier)])
        before = state.snapshot()
        sequence = state.sequence
        context = state.context()
        queue = state.subscribe()
        queue.get_nowait()
        with pytest.raises(ValueError, match="matching content"):
            state.publish(messages)
        await asyncio.sleep(0)
        assert queue.empty()
        assert not state.pending
        assert state.snapshot() == before
        assert state.context() == context
        assert state.sequence == sequence
        state.unsubscribe(queue)
        reconnected = state.subscribe()
        assert reconnected.get_nowait() == before
        state.unsubscribe(reconnected)

    asyncio.run(run())


def test_complete_creation_and_replacement_always_reconnect_with_content():
    async def run():
        state = UIState()
        ui = UI(state)
        data = [{"type": "bar", "x": ["A"], "y": [1]}]
        chart = ui.plot(data, surface_id="sales")
        ui.plot(data, surface_id="other")
        chart.pin()
        ui.plot(data, surface_id="sales", layout={"title": {"text": "Replacement"}})
        queue = state.subscribe()
        snapshot = queue.get_nowait()
        assert [event["messages"][0]["createSurface"]["surfaceId"] for event in snapshot["events"]] == ["other", "sales"]
        assert snapshot["placements"]["sales"] == {"pinned": False}
        for event in snapshot["events"]:
            first, second = event["messages"]
            assert first["createSurface"]["surfaceId"] == second["updateComponents"]["surfaceId"]
        assert snapshot["events"][1]["messages"][1]["updateComponents"]["components"][0]["layout"]["title"]["text"] == "Replacement"
        state.unsubscribe(queue)

    asyncio.run(run())


@pytest.mark.parametrize("resubscribe", [False, True])
def test_unsubscribe_before_delivery_does_not_deliver_to_detached_queue(resubscribe):
    async def run():
        state = UIState()
        detached = state.subscribe()
        detached.get_nowait()
        state.publish([creation("sales"), content("sales")])
        state.unsubscribe(detached)
        if resubscribe:
            replacement = state.subscribe()
            state.publish([content("sales", "Latest")])
        await asyncio.sleep(0)
        assert detached.empty()
        assert not state.pending
        if resubscribe:
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
        assert second.get_nowait() is shared
        public = state.snapshot()
        public["events"][0]["messages"].clear()
        public["events"].clear()
        public["placements"]["sales"]["pinned"] = True
        assert state.snapshot() == shared
        assert len(shared["events"][0]["messages"]) == 2
        state.unsubscribe(first)
        state.unsubscribe(second)

    asyncio.run(run())
