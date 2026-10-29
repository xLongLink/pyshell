import contextlib
import io
import unittest
from unittest.mock import Mock, patch

from pyshell import Agent


class WarmupTests(unittest.TestCase):
    def test_warmup_uses_same_client_without_changing_conversation_or_executing(self):
        client = Mock()
        client.responses.create.return_value.output_text = "raise RuntimeError('never execute')"
        agent = Agent(client, "test-model", service_tier="priority")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with patch("pyshell.time.perf_counter", side_effect=[10.0, 11.0]):
                agent.warm_up()
        client.responses.create.assert_called_once_with(
            model="test-model",
            service_tier="priority",
            reasoning={"effort": "low"},
            input="Reply with only 0.",
            max_output_tokens=16,
            store=False,
        )
        self.assertIsNone(agent.previous_response_id)
        self.assertIsNone(agent.pending_feedback)
        self.assertIn("1.000s", stderr.getvalue())

    def test_failure_is_reported_without_raising(self):
        client = Mock()
        client.responses.create.side_effect = ConnectionError("disconnected")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            Agent(client, "test-model").warm_up()
        self.assertIn("Warm-up failed: disconnected", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
