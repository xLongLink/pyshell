"""Stream subprocess output to a log and console with bounded interruption cleanup."""

import subprocess


def run_logged(command, log_path, *, cwd, on_line=None):
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                if on_line is not None:
                    on_line(line)
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
