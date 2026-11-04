"""Exercise server readiness and cleanup using small local subprocesses."""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from model.benchmark.src.inference_server import inference_server


class TestInferenceServer:
    def command(self, ready):
        source = "import os,time; print(os.getpid(), flush=True); "
        if ready:
            source += "print('listening on http://127.0.0.1:9999', flush=True); "
        return [sys.executable, "-u", "-c", source + "time.sleep(30)"]

    def assert_stopped(self, log):
        pid = int(log.read_text().splitlines()[0])
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    def test_server_is_stopped_when_the_consumer_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            with pytest.raises(ValueError, match="consumer failed"):
                with inference_server(self.command(True), log, port=9999):
                    raise ValueError("consumer failed")
            self.assert_stopped(log)

    def test_timeout_stops_the_server(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            with pytest.raises(RuntimeError, match="did not become ready"):
                with inference_server(self.command(False), log, port=9999, timeout=0.5):
                    pytest.fail("An unready server was accepted")
            self.assert_stopped(log)

    def test_startup_exit_is_not_mistaken_for_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(RuntimeError, match="did not become ready"):
                with inference_server([sys.executable, "-c", "raise SystemExit(3)"], Path(directory) / "server.log", port=9999):
                    pytest.fail("An exited server was accepted")
