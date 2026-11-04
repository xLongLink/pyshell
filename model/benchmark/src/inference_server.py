"""Own a local inference server's log capture, readiness, and shutdown."""

import threading
import subprocess
from pathlib import Path
from contextlib import contextmanager


def gguf_command(local: Path, model: Path, *, port: int, context_size: int, batch_threads: int) -> list[str]:
    """Build the shared CPU-only llama.cpp launch command."""
    return [
        str(local / "llama-b12038/llama-b12038/llama-server"),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context_size),
        "--parallel",
        "1",
        "--threads",
        "8",
        "--threads-batch",
        str(batch_threads),
        "--n-gpu-layers",
        "0",
    ]


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
