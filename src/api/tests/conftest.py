import pytest
from api.ui.session import UI
from src.api.tests.ui_helpers import RecordingUIState


@pytest.fixture
def ui():
    return UI(RecordingUIState())


@pytest.fixture
def country_geography(monkeypatch):
    monkeypatch.setattr(
        "api.ui.plot.choropleth.geography_info", lambda name: {"level": "country", "locations": {"CHE": "Switzerland", "FRA": "France"}}
    )
