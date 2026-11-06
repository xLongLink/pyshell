import io
import os
import sys
import json
import pytest
import signal
from unittest.mock import DEFAULT, call, patch
from importlib.util import find_spec
from model.benchmark.src import worker


@pytest.mark.skipif(
    not (find_spec("api") and find_spec("numpy") and find_spec("pandas")), reason="Install the execution API and benchmark data extra"
)
class TestWorker:
    def run_worker(self, requests, protocol=None):
        incoming = io.StringIO("".join((json.dumps(request) if isinstance(request, dict) else request) + "\n" for request in requests))
        protocol = io.StringIO() if protocol is None else protocol
        errors = io.StringIO()
        with patch.object(sys, "stdin", incoming), patch.object(sys, "stdout", protocol), patch.object(sys, "stderr", errors):
            worker.main()
            assert sys.stdout is protocol
            assert sys.stderr is errors

        assert errors.getvalue() == ""
        lines = protocol.getvalue().splitlines()
        assert len(lines) == len(requests)
        return [json.loads(line) for line in lines]

    @pytest.mark.parametrize("requests", [[], [{"action": "reset", "source": "pass"}]])
    def test_previous_signal_handler_is_restored_on_eof(self, requests):
        def previous_handler(_signal, _frame):
            pytest.fail("Previous signal handler should not run")

        timer = signal.setitimer(signal.ITIMER_REAL, 0)
        handler = signal.signal(signal.SIGALRM, previous_handler)
        try:
            signal.setitimer(signal.ITIMER_REAL, 60)
            self.run_worker(requests)
            assert signal.getsignal(signal.SIGALRM) is previous_handler
            assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, *timer)

    @pytest.mark.parametrize("failure_point", ["open", "publish"])
    def test_previous_signal_handler_is_restored_on_escaping_failure(self, failure_point):
        def previous_handler(_signal, _frame):
            pytest.fail("Previous signal handler should not run")

        failure = OSError("fatal failure")
        protocol = io.StringIO()
        timer = signal.setitimer(signal.ITIMER_REAL, 0)
        handler = signal.signal(signal.SIGALRM, previous_handler)
        try:
            signal.setitimer(signal.ITIMER_REAL, 60)
            with (
                patch.object(worker, "open", wraps=open, side_effect=failure if failure_point == "open" else None, create=True),
                patch.object(
                    protocol,
                    "write",
                    wraps=protocol.write,
                    side_effect=[ValueError("first publish failed"), failure] if failure_point == "publish" else None,
                ),
                pytest.raises(OSError, match="fatal failure") as raised,
            ):
                self.run_worker([{"action": "reset", "source": "pass"}], protocol=protocol)
            assert raised.value is failure
            assert signal.getsignal(signal.SIGALRM) is previous_handler
            assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, *timer)

    def test_sink_is_opened_once_and_closed_after_requests(self):
        sink = open(os.devnull, "w")
        try:
            with patch.object(worker, "open", return_value=sink, create=True) as opened:
                results = self.run_worker([{"action": "reset", "source": "pass"}, {"action": "execute", "source": "pass"}])
            opened.assert_called_once_with(os.devnull, "w")
            assert all(result["exception"] is None for result in results)
            assert sink.closed
        finally:
            sink.close()

    @pytest.mark.parametrize(
        "message,alarms",
        [
            ({"action": "reset", "source": "pass"}, [10, 0]),
            ({"action": "reset", "source": "raise ValueError('failed')"}, [10, 0]),
            ({"action": "reset", "source": "import signal; signal.raise_signal(signal.SIGALRM)"}, [10, 0]),
            ("not json", [0]),
            ("null", [0]),
            ({"action": "execute"}, [10, 0]),
            ({"action": "unknown", "source": "pass"}, [0]),
        ],
    )
    def test_request_timer_is_armed_and_cleared(self, message, alarms):
        with patch.object(signal, "alarm", wraps=signal.alarm) as alarm:
            self.run_worker([message])
        assert alarm.call_args_list == [call(seconds) for seconds in alarms]

    def test_output_and_exception_remain_in_json_feedback(self):
        results = self.run_worker(
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

    def test_serialization_failure_returns_error_and_keeps_the_workspace(self):
        dumps = json.dumps

        def encode(result, *args, **kwargs):
            if result.get("stdout") == "first\n":
                raise ValueError("serialization failed")
            return dumps(result, *args, **kwargs)

        with patch.object(worker.json, "dumps", side_effect=encode):
            results = self.run_worker(
                [{"action": "reset", "source": "value = 42\nprint('first')"}, {"action": "execute", "source": "print(value)"}]
            )
        assert results[0]["exception"] == "ValueError: serialization failed"
        assert results[1]["exception"] is None
        assert results[1]["stdout"] == "42\n"

    def test_write_failure_returns_error_and_keeps_the_workspace(self):
        protocol = io.StringIO()
        with patch.object(protocol, "write", wraps=protocol.write, side_effect=[OSError("write failed"), DEFAULT, DEFAULT]):
            results = self.run_worker(
                [{"action": "reset", "source": "value = 42"}, {"action": "execute", "source": "print(value)"}], protocol=protocol
            )
        assert results[0]["exception"] == "OSError: write failed"
        assert results[1]["exception"] is None
        assert results[1]["stdout"] == "42\n"

    def test_malformed_requests_do_not_break_state_or_protocol(self):
        results = self.run_worker(
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

    def test_consecutive_requests_share_sink_and_reset_namespace(self):
        results = self.run_worker(
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
        results = self.run_worker(
            [
                {"action": "reset", "source": "import signal\nvalue = 9\nsignal.raise_signal(signal.SIGALRM)"},
                {"action": "execute", "source": "print(value)"},
            ]
        )

        assert "TimeoutError: Execution exceeded ten seconds" in results[0]["exception"]
        assert results[0]["done"] is None
        assert results[1] == {"stdout": "9\n", "stderr": "", "exception": None, "done": None}

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
        results = self.run_worker([{"action": "reset", "source": source}, {"action": "execute", "source": "print('still ready')"}])

        assert results[0]["exception"] is None
        assert int(results[0]["stdout"]) < 1048576
        assert results[0]["stderr"] == ""
        assert results[1] == {"stdout": "still ready\n", "stderr": "", "exception": None, "done": None}

    @pytest.mark.parametrize("action", ["unknown", "", None, []])
    def test_unsupported_actions_do_not_execute_source_or_change_workspace(self, action):
        results = self.run_worker(
            [
                {"action": "reset", "source": "value = 7"},
                {"action": action, "source": "value = 99"},
                {"action": "execute", "source": "print(value)"},
            ]
        )
        assert results[1] == {"stdout": "", "stderr": "", "exception": "ValueError: Unsupported worker action", "done": None}
        assert results[2] == {"stdout": "7\n", "stderr": "", "exception": None, "done": None}
