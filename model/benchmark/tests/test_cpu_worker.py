"""Exercise the worker protocol with the real execution shell."""

import io
import os
import sys
import json
import pytest
import signal
from unittest.mock import DEFAULT, call, patch
from importlib.util import find_spec
from model.benchmark.src import cpu_worker


@pytest.mark.skipif(
    not (find_spec("api") and find_spec("numpy") and find_spec("pandas")), reason="Install the execution API and benchmark data extra"
)
class TestCPUWorker:
    def run_worker(self, requests, protocol=None):
        incoming = io.StringIO("".join((json.dumps(request) if isinstance(request, dict) else request) + "\n" for request in requests))
        protocol = io.StringIO() if protocol is None else protocol
        errors = io.StringIO()
        sinks = []

        def open_sink(*args, **kwargs):
            sink = open(*args, **kwargs)
            sinks.append(sink)
            return sink

        handler = signal.getsignal(signal.SIGALRM)
        try:
            with (
                patch.object(sys, "stdin", incoming),
                patch.object(sys, "stdout", protocol),
                patch.object(sys, "stderr", errors),
                patch.object(cpu_worker, "open", side_effect=open_sink, create=True) as opened,
                patch.object(signal, "alarm", wraps=signal.alarm) as alarm,
            ):
                cpu_worker.main([])
                assert sys.stdout is protocol
                assert sys.stderr is errors
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, handler)

        opened.assert_called_once_with(os.devnull, "w")
        assert sinks[0].closed
        assert errors.getvalue() == ""
        lines = protocol.getvalue().splitlines()
        assert len(lines) == len(requests)
        return [json.loads(line) for line in lines], alarm.call_args_list

    def test_output_and_exception_remain_in_json_feedback(self):
        results, alarms = self.run_worker(
            [
                {"action": "reset", "source": "import sys\nprint('hello')\nprint('warning', file=sys.stderr)\ndone('finished')"},
                {
                    "action": "execute",
                    "source": (
                        "value = 42\nprint('before failure')\nprint('error detail', file=sys.stderr)\n"
                        "done('invalid')\nraise ValueError('failed')"
                    ),
                },
                {"action": "execute", "source": "print(value)"},
            ]
        )

        assert results[0] == {"stdout": "hello\n", "stderr": "warning\n", "exception": None, "done": "finished"}
        assert results[1]["stdout"] == "before failure\n"
        assert results[1]["stderr"] == "error detail\n"
        assert "ValueError: failed" in results[1]["exception"]
        assert results[1]["done"] is None
        assert results[2] == {"stdout": "42\n", "stderr": "", "exception": None, "done": None}
        assert alarms == [call(10), call(0)] * 3

    def test_serialization_failure_returns_error_and_keeps_the_workspace(self):
        dumps = json.dumps

        def encode(result, *args, **kwargs):
            if result.get("stdout") == "first\n":
                raise ValueError("serialization failed")
            return dumps(result, *args, **kwargs)

        with patch.object(cpu_worker.json, "dumps", side_effect=encode):
            results, _ = self.run_worker(
                [{"action": "reset", "source": "value = 42\nprint('first')"}, {"action": "execute", "source": "print(value)"}]
            )
        assert results[0]["exception"] == "ValueError: serialization failed"
        assert results[1]["exception"] is None
        assert results[1]["stdout"] == "42\n"

    def test_write_failure_returns_error_and_keeps_the_workspace(self):
        protocol = io.StringIO()
        with patch.object(protocol, "write", wraps=protocol.write, side_effect=[OSError("write failed"), DEFAULT, DEFAULT]):
            results, _ = self.run_worker(
                [{"action": "reset", "source": "value = 42"}, {"action": "execute", "source": "print(value)"}], protocol=protocol
            )
        assert results[0]["exception"] == "OSError: write failed"
        assert results[1]["exception"] is None
        assert results[1]["stdout"] == "42\n"

    def test_malformed_requests_do_not_break_state_or_protocol(self):
        results, alarms = self.run_worker(
            [
                {"action": "reset", "source": "value = 7"},
                "not json",
                {"action": "execute"},
                "null",
                {"action": "execute", "source": "print(value)"},
            ]
        )

        for result, error in zip(results[1:4], ("JSONDecodeError:", "KeyError:", "TypeError:"), strict=True):
            assert result["stdout"] == ""
            assert result["stderr"] == ""
            assert result["exception"].startswith(error)
            assert result["done"] is None
        assert results[-1] == {"stdout": "7\n", "stderr": "", "exception": None, "done": None}
        assert alarms == [call(10), call(0), call(0), call(10), call(0), call(10), call(0), call(10), call(0)]

    def test_consecutive_requests_share_sink_and_reset_namespace(self):
        results, _ = self.run_worker(
            [
                {"action": "reset", "source": "import sys\nsink = sys.stdout.stream\nvalue = 7"},
                {
                    "action": "execute",
                    "source": (
                        "assert sys.stdout.stream is sink\nassert sys.stderr.stream is sink\n"
                        "assert not sink.closed\nvalue += 1\nprint(value)"
                    ),
                },
                {
                    "action": "reset",
                    "source": (
                        "import sys\nassert sys.stdout.stream is sys.stderr.stream\nassert not sys.stdout.stream.closed\n"
                        "assert 'value' not in globals()\n"
                        "print(np.arange(3).sum(), pd.Series([1, 2]).sum(), statistics.mean([2, 4]), json.dumps(True))"
                    ),
                },
                {"action": "execute", "source": "assert 'value' not in globals()\nprint('ready')"},
            ]
        )

        assert all(result["exception"] is None for result in results), results
        assert [result["stdout"] for result in results] == ["", "8\n", "3 3 3 true\n", "ready\n"]

    def test_timeout_feedback_allows_next_request(self):
        results, alarms = self.run_worker(
            [
                {"action": "reset", "source": "import signal\nvalue = 9\nsignal.raise_signal(signal.SIGALRM)"},
                {"action": "execute", "source": "print(value)"},
            ]
        )

        assert "TimeoutError: Execution exceeded ten seconds" in results[0]["exception"]
        assert results[0]["done"] is None
        assert results[1] == {"stdout": "9\n", "stderr": "", "exception": None, "done": None}
        assert alarms == [call(10), call(0)] * 2

    def test_discarded_logs_use_bounded_memory(self):
        source = (
            "import sys, tracemalloc\n"
            "tracemalloc.start()\n"
            "for _ in range(512):\n"
            "    sys.stdout.stream.write('x' * 65536)\n"
            "    sys.stderr.stream.write('y' * 65536)\n"
            "_, peak = tracemalloc.get_traced_memory()\n"
            "tracemalloc.stop()\n"
            "print(peak)"
        )
        results, _ = self.run_worker([{"action": "reset", "source": source}, {"action": "execute", "source": "print('still ready')"}])

        assert results[0]["exception"] is None
        assert int(results[0]["stdout"]) < 1048576
        assert results[0]["stderr"] == ""
        assert results[1] == {"stdout": "still ready\n", "stderr": "", "exception": None, "done": None}
