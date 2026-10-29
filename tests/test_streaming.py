import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pyshell import Agent


def stream_for(chunks, status="completed"):
    stream = Mock()
    stream.__enter__ = Mock(return_value=stream)
    stream.__exit__ = Mock(return_value=False)
    stream.__iter__ = Mock(return_value=iter(
        SimpleNamespace(type="response.output_text.delta", delta=chunk)
        for chunk in chunks
    ))
    stream.get_final_response.return_value = SimpleNamespace(
        id="r1", output_text="".join(chunks), status=status,
    )
    return stream


class StreamingTests(unittest.TestCase):
    def test_requests_priority_processing(self):
        client = Mock()
        client.responses.stream.return_value = stream_for(["done()"])
        agent = Agent(client, "test-model", service_tier="priority")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            agent.run("finish")
        self.assertEqual(client.responses.stream.call_args.kwargs["service_tier"], "priority")
        self.assertEqual(client.responses.stream.call_args.kwargs["reasoning"], {"effort": "low"})

    def test_prints_first_token_time_and_duration(self):
        client = Mock()
        client.responses.stream.return_value = stream_for(["", "done()"])
        agent = Agent(client, "test-model")
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            with patch("pyshell.time.perf_counter", side_effect=[10.0, 10.25, 11.5]):
                agent.run("finish")
        self.assertEqual(
            stderr.getvalue(),
            "[Time to first token: 0.250s | Total duration: 1.500s]\n",
        )

    def test_timing_when_stream_fails_before_first_token(self):
        client = Mock()
        client.responses.stream.side_effect = ConnectionError("disconnected")
        agent = Agent(client, "test-model")
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            with patch("pyshell.time.perf_counter", side_effect=[10.0, 12.0]):
                with self.assertRaises(ConnectionError):
                    agent.run("finish")
        self.assertEqual(
            stderr.getvalue(),
            "[Time to first token: n/a | Total duration: 2.000s]\n",
        )

    def test_continues_past_twenty_steps_until_done(self):
        client = Mock()
        client.responses.stream.side_effect = [
            stream_for(["pass"]) for _ in range(21)
        ] + [stream_for(["done()"])]
        agent = Agent(client, "test-model")
        with contextlib.redirect_stdout(io.StringIO()):
            agent.run("continue until finished")
        self.assertEqual(client.responses.stream.call_count, 22)
        self.assertTrue(json.loads(agent.pending_feedback)["done"])

    def test_executes_lines_before_response_finishes(self):
        client = Mock()
        stream = stream_for(['print(', '"hello")\n', 'done()'])
        client.responses.stream.return_value = stream
        agent = Agent(client, "test-model")
        output = io.StringIO()

        def final_response():
            self.assertEqual(output.getvalue(), '\n>>> print("hello")\nhello\n>>> done()')
            self.assertIsNone(agent.pending_feedback)
            return SimpleNamespace(
                id="r1", output_text='print("hello")\ndone()', status="completed",
            )

        stream.get_final_response.side_effect = final_response
        with contextlib.redirect_stdout(output):
            agent.run("say hello")
        self.assertEqual(output.getvalue(), '\n>>> print("hello")\nhello\n>>> done()\n')
        self.assertEqual(json.loads(agent.pending_feedback)["stdout"], "hello\n")
        self.assertEqual(agent.previous_response_id, "r1")

    def test_incomplete_response_is_not_executed(self):
        client = Mock()
        client.responses.stream.return_value = stream_for(["x = 42"], status="incomplete")
        agent = Agent(client, "test-model")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            agent.run("set x")
        self.assertNotIn("x", agent.shell.namespace)
        self.assertIsNone(agent.previous_response_id)

    def test_stream_failure_is_not_executed_and_preserves_feedback(self):
        client = Mock()
        stream = stream_for(["x = 42"])
        stream.get_final_response.side_effect = ConnectionError("disconnected")
        client.responses.stream.return_value = stream
        agent = Agent(client, "test-model")
        agent.pending_feedback = "previous output"
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ConnectionError):
            agent.run("set x")
        self.assertNotIn("x", agent.shell.namespace)
        self.assertTrue(agent.pending_feedback.startswith("previous output\n"))
        self.assertIn("ConnectionError", agent.pending_feedback)

    def test_line_executes_before_next_stream_event(self):
        client = Mock()
        stream = stream_for(["x = 42\n", "done()\n"])
        agent = Agent(client, "test-model")

        def events():
            yield SimpleNamespace(type="response.output_text.delta", delta="x = 42\n")
            self.assertEqual(agent.shell.namespace["x"], 42)
            yield SimpleNamespace(type="response.output_text.delta", delta="done()\n")

        stream.__iter__.side_effect = events
        client.responses.stream.return_value = stream
        with contextlib.redirect_stdout(io.StringIO()):
            agent.run("set x")

    def test_multiline_blocks_and_expression_display(self):
        client = Mock()
        client.responses.stream.return_value = stream_for([
            "values = []\nfor i in range(3):\n",
            "    values.append(i)\n", "\nvalues\ndone()",
        ])
        agent = Agent(client, "test-model")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            agent.run("count")
        self.assertEqual(agent.shell.namespace["values"], [0, 1, 2])
        self.assertEqual(json.loads(agent.pending_feedback)["stdout"], "[0, 1, 2]\n")
        self.assertIn("...     values.append(i)", output.getvalue())

    def test_disconnect_keeps_executed_lines_but_discards_partial_line(self):
        client = Mock()
        stream = stream_for(["x = 42\ny = 99"])
        stream.get_final_response.side_effect = ConnectionError("disconnected")
        client.responses.stream.return_value = stream
        agent = Agent(client, "test-model")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ConnectionError):
            agent.run("set variables")
        self.assertEqual(agent.shell.namespace["x"], 42)
        self.assertNotIn("y", agent.shell.namespace)

    def test_error_output_is_sent_back_and_next_line_runs(self):
        client = Mock()
        client.responses.stream.return_value = stream_for(["1 / 0\nprint('ok')\ndone()"])
        agent = Agent(client, "test-model")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            agent.run("test errors")
        result = json.loads(agent.pending_feedback)
        self.assertIn("ZeroDivisionError", result["exception"])
        self.assertEqual(result["stdout"], "ok\n")


if __name__ == "__main__":
    unittest.main()
