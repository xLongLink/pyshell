"""Own a local inference server's log capture, readiness, and shutdown."""

import threading
import subprocess
from contextlib import contextmanager


@contextmanager
def inference_server(command, log_path, port, timeout=90):
    ready = threading.Event()
    listening = False
    with log_path.open("w") as log:
        server = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        def capture():
            nonlocal listening
            try:
                for line in server.stdout:
                    log.write(line)
                    log.flush()
                    if f"listening on http://127.0.0.1:{port}" in line:
                        listening = True
                        ready.set()
            finally:
                ready.set()

        thread = threading.Thread(target=capture, daemon=True)
        thread.start()
        try:
            if not ready.wait(timeout) or not listening or server.poll() is not None:
                raise RuntimeError(f"Inference server did not become ready; inspect {log_path}")
            yield
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            thread.join(timeout=5)
            server.stdout.close()
