import httpx
import pytest
import asyncio
from sample.src.shell import RemoteShell


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
        return httpx.Response(200, json=payload)

    async def run():
        async with httpx.AsyncClient(
            base_url="http://execution", transport=httpx.MockTransport(handle), timeout=httpx.Timeout(connect=1, read=2, write=3, pool=4)
        ) as client:
            shell = RemoteShell(client)
            assert await getattr(shell, method)() == payload
            assert shell.uncertain is False

    asyncio.run(run())

    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.path == path
    assert requests[0].extensions["timeout"] == expected_timeout


@pytest.mark.parametrize("method", ["inventory", "context"])
@pytest.mark.parametrize("status", [404, 503])
def test_metadata_reads_raise_for_unsuccessful_status_without_marking_execution_uncertain(method, status):
    async def run():
        transport = httpx.MockTransport(lambda request: httpx.Response(status, json={"detail": "unavailable"}))
        async with httpx.AsyncClient(base_url="http://execution", transport=transport) as client:
            shell = RemoteShell(client)
            with pytest.raises(httpx.HTTPStatusError) as error:
                await getattr(shell, method)()
            assert error.value.response.status_code == status
            assert shell.uncertain is False

    asyncio.run(run())
