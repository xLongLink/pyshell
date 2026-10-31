import json
import httpx2
import pytest
import asyncio
from sample.src.shell import RemoteShell


@pytest.mark.parametrize(
    "payload",
    [
        '{"type":"execution_finished","extra":true}\n',
        '{"type":"result","stdout":"","stderr":"","exception":null,"done":true,"extra":true}\n',
        '{"type":"result","stdout":"","stderr":"","exception":null,"done":"true"}\n',
        '{"type":"result"}\n',
        '{"type":"unknown"}\n',
        "not json\n",
    ],
)
def test_malformed_stream_marks_execution_uncertain_and_prevents_resubmission(payload):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx2.Response(200, text=payload)

    async def run():
        async with httpx2.AsyncClient(base_url="http://execution", transport=httpx2.MockTransport(handle)) as client:
            shell = RemoteShell(client)
            with pytest.raises(RuntimeError, match="Execution was not confirmed"):
                await shell.stream("print('once')")
            assert shell.uncertain
            with pytest.raises(RuntimeError, match="Execution state is uncertain"):
                await shell.stream("print('once')")

    asyncio.run(run())
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("method", "path", "payload", "expected_timeout"),
    [
        ("inventory", "/libraries", [{"name": "numpy", "version": "2.0"}], {"connect": 1, "read": 2, "write": 3, "pool": 4}),
        ("context", "/ui/context", {"charts": []}, {"connect": 5, "read": 5, "write": 5, "pool": 5}),
    ],
)
def test_metadata_reads_preserve_payload_path_and_timeout(method, path, payload, expected_timeout):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx2.Response(200, json=payload)

    async def run():
        async with httpx2.AsyncClient(
            base_url="http://execution", transport=httpx2.MockTransport(handle), timeout=httpx2.Timeout(connect=1, read=2, write=3, pool=4)
        ) as client:
            shell = RemoteShell(client)
            assert await getattr(shell, method)() == payload
            assert shell.uncertain is False

    asyncio.run(run())

    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.path == path
    assert requests[0].extensions["timeout"] == expected_timeout


@pytest.mark.parametrize(
    ("events", "confirmed"),
    [
        (["result", "execution_finished"], True),
        ([], False),
        (["result"], False),
        (["execution_finished"], False),
        (["result", "result", "execution_finished"], False),
        (["result", "execution_finished", "execution_finished"], False),
        (["result", "execution_finished", "result"], False),
    ],
)
def test_stream_requires_one_result_followed_by_one_terminal_marker(events, confirmed):
    result = {"type": "result", "stdout": "finished", "stderr": "", "exception": None, "done": True}
    payload = "".join(json.dumps(result if event == "result" else {"type": event}) + "\n" for event in events)

    async def run():
        transport = httpx2.MockTransport(lambda request: httpx2.Response(200, text=payload))
        async with httpx2.AsyncClient(base_url="http://execution", transport=transport) as client:
            shell = RemoteShell(client)
            if confirmed:
                assert (await shell.stream("done()")).model_dump() == result
                assert not shell.uncertain
            else:
                with pytest.raises(RuntimeError, match="Execution was not confirmed"):
                    await shell.stream("done()")
                assert shell.uncertain
                with pytest.raises(RuntimeError, match="Execution state is uncertain"):
                    await shell.stream("done()")

    asyncio.run(run())


@pytest.mark.parametrize(("method", "status"), [("inventory", 404), ("context", 503)])
def test_metadata_reads_raise_for_unsuccessful_status_without_marking_execution_uncertain(method, status):
    async def run():
        transport = httpx2.MockTransport(lambda request: httpx2.Response(status, json={"detail": "unavailable"}))
        async with httpx2.AsyncClient(base_url="http://execution", transport=transport) as client:
            shell = RemoteShell(client)
            with pytest.raises(httpx2.HTTPStatusError) as error:
                await getattr(shell, method)()
            assert error.value.response.status_code == status
            assert shell.uncertain is False

    asyncio.run(run())
