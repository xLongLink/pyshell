import pytest


@pytest.fixture
def country_geography(monkeypatch):
    monkeypatch.setattr(
        "api.ui.plot.choropleth._geography_info", lambda name: {"level": "country", "locations": {"CHE": "Switzerland", "FRA": "France"}}
    )
