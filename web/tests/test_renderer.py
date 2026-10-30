import httpx
import pytest
import asyncio
from types import SimpleNamespace
from web.main import app
from web.src.shell import RemoteShell
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    ("path", "content_type", "content"),
    [
        ("", "text/html; charset=utf-8", b"<!doctype html><title>Renderer</title>"),
        ("assets/renderer-abc.js", "text/javascript", b"console.log('renderer')"),
        ("assets/style-abc.css", "text/css; charset=utf-8", b"body { margin: 0; }"),
    ],
)
def test_renderer_proxy_preserves_assets_and_supports_sandbox_fetches(monkeypatch, path, content_type, content):
    requests = []

    def respond(request):
        requests.append(request.url.path)
        return httpx.Response(200, content=content, headers={"content-type": content_type})

    async def run():
        async with httpx.AsyncClient(base_url="http://execution", transport=httpx.MockTransport(respond)) as client:
            shell = RemoteShell(client)
            monkeypatch.setattr(app.state, "workspace", SimpleNamespace(agent=SimpleNamespace(shell=shell)), raising=False)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver") as browser:
                response = await browser.get(f"/renderer/{path}")
            assert response.status_code == 200
            assert response.content == content
            assert response.headers["content-type"] == content_type
            assert response.headers["access-control-allow-origin"] == "*"
            assert response.headers["cache-control"] == "no-store"
            assert not shell.uncertain

    asyncio.run(run())
    assert requests == [f"/renderer/{path}"]


@pytest.mark.parametrize(("status", "expected"), [(404, 404), (500, 502), (None, 502)])
def test_renderer_proxy_handles_upstream_failures(monkeypatch, status, expected):
    async def renderer(path):
        request = httpx.Request("GET", "http://private/renderer/")
        if status is None:
            raise httpx.ConnectError("private backend", request=request)
        httpx.Response(status, request=request).raise_for_status()

    monkeypatch.setattr(
        app.state, "workspace", SimpleNamespace(agent=SimpleNamespace(shell=SimpleNamespace(renderer=renderer))), raising=False
    )
    response = TestClient(app).get("/renderer/")
    assert response.status_code == expected
    assert "private" not in response.text


def test_renderer_proxy_rejects_encoded_traversal():
    response = TestClient(app).get("/renderer/%2e%2e/health")
    assert response.status_code == 404
