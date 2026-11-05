"""Check the reference executor's persistent state and completion semantics."""

import sys
import pytest
from model.benchmark.src.reference import feedback


class TestReference:
    def test_completion_accepts_message_keyword_and_returns_none(self):
        result = feedback("returned = done(message='first')\nassert returned is None\ndone(message='last')", {})
        assert result["exception"] is None
        assert result["done"] == "last"

    @pytest.mark.parametrize("message", [None, False, 0, "", "   ", [], {}, [1], {"value": 2}])
    def test_last_completion_preserves_unvalidated_values(self, message):
        result = feedback("done('first')\ndone(message)", {"message": message})
        assert result["exception"] is None
        assert result["done"] is message

    def test_completion_retains_mutable_value_without_copying(self):
        message = [1]
        result = feedback("done('first')\ndone(message)\nmessage.append(2)", {"message": message})
        assert result["exception"] is None
        assert result["done"] is message
        assert result["done"] == [1, 2]

    def test_captures_both_streams_and_completion(self):
        result = feedback("print('output')\nprint('error', file=sys.stderr)\ndone('finished')", {"sys": sys})
        assert result == {"stdout": "output\n", "stderr": "error\n", "exception": None, "done": "finished"}

    def test_partial_updates_survive_errors_but_completion_does_not(self):
        namespace = {"values": [1, 2]}
        result = feedback("values.append(3)\ndone('premature')\nraise ValueError('failed')", namespace)
        assert namespace["values"] == [1, 2, 3]
        assert result["done"] is None
        assert "ValueError: failed" in result["exception"]

    def test_completion_is_not_carried_to_the_next_submission(self):
        namespace = {}
        assert feedback("done('first')", namespace)["done"] == "first"
        assert feedback("answer = 42", namespace)["done"] is None
