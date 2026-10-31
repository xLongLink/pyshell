import httpx2
import pytest
import asyncio
from sample.src.shell import RemoteShell

RESULT = '{"type":"result","stdout":"finished","stderr":"","exception":null,"done":true}\n'
FINISHED = '{"type":"execution_finished"}\n'


@pytest.mark.parametrize(
    ("payload", "confirmed"),
    [
        ('{"type":"execution_finished","extra":true}\n', False),
        ('{"type":"result","stdout":"","stderr":"","exception":null,"done":true,"extra":true}\n', False),
        ('{"type":"result","stdout":"","stderr":"","exception":null,"done":"true"}\n', False),
        ('{"type":"result"}\n', False),
        ('{"type":"unknown"}\n', False),
        ("not json\n", False),
        (RESULT + FINISHED, True),
        ("", False),
        (RESULT, False),
        (FINISHED, False),
        (FINISHED + RESULT, False),
        (RESULT + RESULT, False),
        (FINISHED + FINISHED, False),
        (RESULT + RESULT + FINISHED, False),
        (RESULT + FINISHED + FINISHED, False),
        (RESULT + FINISHED + RESULT, False),
    ],
)
def test_stream_requires_one_valid_result_followed_by_one_terminal_marker(payload, confirmed):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx2.Response(200, text=payload)

    async def run():
        async with httpx2.AsyncClient(base_url="http://execution", transport=httpx2.MockTransport(handle)) as client:
            shell = RemoteShell(client)
            if confirmed:
                assert (await shell.stream("print('once')")).model_dump() == {
                    "type": "result",
                    "stdout": "finished",
                    "stderr": "",
                    "exception": None,
                    "done": True,
                }
                assert not shell.uncertain
            else:
                with pytest.raises(RuntimeError, match="Execution was not confirmed"):
                    await shell.stream("print('once')")
                assert shell.uncertain
                with pytest.raises(RuntimeError, match="Execution state is uncertain"):
                    await shell.stream("print('once')")

    asyncio.run(run())
    assert len(requests) == 1


def test_inventory_caches_empty_successful_reads():
    payload = []
    requests = []

    def handle(request):
        requests.append(request)
        return httpx2.Response(200, json=payload)

    async def run():
        async with httpx2.AsyncClient(base_url="http://execution", transport=httpx2.MockTransport(handle)) as client:
            shell = RemoteShell(client)
            assert await shell.inventory() == payload
            assert await shell.inventory() == payload

    asyncio.run(run())
    assert len(requests) == 1


@pytest.mark.parametrize("failure", ["status", "transport", "json"])
def test_inventory_retries_after_failed_reads(failure):
    requests = []
    payload = [{"name": "numpy", "version": "2.0"}]

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            if failure == "transport":
                raise httpx2.ConnectError("unavailable", request=request)
            if failure == "json":
                return httpx2.Response(200, text="invalid json")
            return httpx2.Response(503)
        return httpx2.Response(200, json=payload)

    async def run():
        async with httpx2.AsyncClient(base_url="http://execution", transport=httpx2.MockTransport(handle)) as client:
            shell = RemoteShell(client)
            with pytest.raises((httpx2.HTTPError, ValueError)):
                await shell.inventory()
            assert await shell.inventory() == payload
            assert await shell.inventory() == payload
            assert not shell.uncertain

    asyncio.run(run())
    assert len(requests) == 2
