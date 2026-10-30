from main import create_app
from fastapi.testclient import TestClient


def test_renderer_serves_built_assets_without_masking_missing_files(tmp_path):
    (tmp_path / "index.html").write_text('<script type="module" src="./assets/app.js"></script>')
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("console.log('renderer')")
    app = create_app()
    renderer = next(route.app for route in app.routes if route.path == "/renderer")
    renderer.directory = str(tmp_path)
    renderer.all_directories = [str(tmp_path)]
    with TestClient(app) as client:
        response = client.get("/renderer/")
        assert response.status_code == 200
        assert "./assets/app.js" in response.text
        asset = client.get("/renderer/assets/app.js")
        assert asset.status_code == 200
        assert "javascript" in asset.headers["content-type"]
        assert client.get("/renderer/missing.js").status_code == 404
        assert client.get("/health").json() == {"status": "ok"}
