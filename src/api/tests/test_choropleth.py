import pytest
from pydantic import ValidationError
from api.ui.plot import geography
from urllib.error import URLError
from api.ui.session import UI

INVALID_TRACES = [
    pytest.param({"locations": ["CH-ZH"], "values": [float("inf")]}, id="nonfinite-value"),
    pytest.param({"locations": ["CH-ZH"], "values": []}, id="mismatched-lengths"),
    pytest.param({"locations": ["CH-ZH", "CH-ZH"], "values": [1, 2]}, id="duplicate-locations"),
    pytest.param({"level": "country", "locations": ["CH-ZH"], "values": [1]}, id="incompatible-level"),
]


@pytest.mark.parametrize("trace", INVALID_TRACES)
def test_validation_checks_input_structure_before_fetching(monkeypatch, trace):
    # Arrange
    def unexpected_download(url):
        pytest.fail("Invalid input must be rejected before fetching geography")

    monkeypatch.setattr(geography, "download", unexpected_download)

    # Act
    # Assert
    with pytest.raises(ValidationError):
        UI().plot([{"type": "choropleth", "geography": "CHE", "level": "region", **trace}])


def test_failed_first_lookup_preserves_an_existing_plot(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path))
    ui = UI()
    chart = ui.plot([{"type": "bar", "x": ["A"], "y": [1]}])
    messages = list(ui.messages)

    def unavailable(url):
        raise URLError("offline")

    monkeypatch.setattr(geography, "download", unavailable)
    with pytest.raises(ValidationError, match="first use requires internet"):
        chart.update(data=[{"type": "choropleth", "geography": "CHE", "level": "region", "locations": ["CH-ZH"], "values": [1]}])
    assert chart.data[0].type == "bar"
    assert ui.messages == messages
