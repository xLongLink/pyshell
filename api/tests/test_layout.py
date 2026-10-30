import pytest
from pydantic import ValidationError
from src.ui.plot.layout import Layout


def test_layout_supplies_default_titles_axes_and_legend():

    # Arrange / Act
    layout = Layout()

    # Assert
    assert layout.model_dump() == {"title": {"text": ""}, "xaxis": {"title": {"text": ""}}, "showlegend": True}


def test_layout_preserves_explicit_titles_and_hidden_legend():

    # Arrange
    data = {"title": {"text": "Revenue"}, "xaxis": {"title": {"text": "Month"}}, "showlegend": False}

    # Act
    layout = Layout.model_validate(data)

    # Assert
    assert layout.model_dump() == data


@pytest.mark.parametrize("data", [{"url": "https://example.com"}, {"xaxis": {"range": [0, 10]}}, {"yaxis": {"title": {"text": "Amount"}}}])
def test_layout_rejects_unsupported_options(data):

    # Act / Assert
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Layout.model_validate(data)
