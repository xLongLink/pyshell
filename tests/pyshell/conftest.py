import pytest
from types import SimpleNamespace
from unittest.mock import Mock


@pytest.fixture
def stream_for():
    def build(chunks, status="completed"):
        stream = Mock()
        stream.__enter__ = Mock(return_value=stream)
        stream.__exit__ = Mock(return_value=False)
        stream.__iter__ = Mock(return_value=iter(
            SimpleNamespace(type="response.output_text.delta", delta=chunk)
            for chunk in chunks
        ))

        # The final response contains the same text as the streamed deltas.
        stream.get_final_response.return_value = SimpleNamespace(
            id="r1", output_text="".join(chunks), status=status,
        )
        return stream

    return build
