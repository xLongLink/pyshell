from src.ui.plot.layout import Layout


def test_layout_preserves_explicit_titles_and_hidden_legend():

    # Arrange
    data = {"title": {"text": "Revenue"}, "xaxis": {"title": {"text": "Month"}}, "showlegend": False}

    # Act
    layout = Layout.model_validate(data)

    # Assert
    assert layout.model_dump() == data
