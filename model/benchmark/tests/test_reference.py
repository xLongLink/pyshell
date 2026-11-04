"""Check the reference executor's persistent state and completion semantics."""

import sys
from model.benchmark.src.reference import feedback


class TestReference:
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
