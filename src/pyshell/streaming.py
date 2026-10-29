import sys
import time


def stream_response(client, shell, kwargs):
    shell.begin_response()
    started = time.perf_counter()
    first_token = None
    line = ""
    at_line_start = True
    wrote_text = False
    print()

    # The stream context must close even if local execution is interrupted.
    try:
        with client.responses.stream(**kwargs) as stream:

            # Ignore metadata and reasoning events; only text is executable.
            for event in stream:
                if event.type != "response.output_text.delta":
                    continue

                # Empty deltas do not count as the first code token.
                if event.delta and first_token is None:
                    first_token = time.perf_counter() - started

                # Deltas can split a line or contain several complete lines.
                for part in event.delta.splitlines(keepends=True):

                    # Continuation prompts follow the interactive console's buffer.
                    if at_line_start:
                        print("... " if shell.buffer else ">>> ", end="", flush=True)

                    # Echo code before executing it, preserving stream order.
                    print(part, end="", flush=True)
                    at_line_start = part.endswith("\n")
                    wrote_text = True
                    line += part

                    # Submit only complete lines while the stream is active.
                    if at_line_start:
                        shell.execute_line(line.removesuffix("\n").removesuffix("\r"))
                        line = ""

            # Failed responses must not execute their unterminated final line.
            response = stream.get_final_response()
            if response.status != "completed":
                raise RuntimeError(
                    f"Model response {response.status}; remaining input discarded. "
                    "Previously executed lines remain in effect."
                )

            # A successful response may omit its final newline.
            if line:
                print(flush=True)
                at_line_start = True
                shell.execute_line(line)

            # A blank line completes compound statements in InteractiveConsole.
            if shell.buffer:
                print("... ", flush=True)
                shell.execute_line("")

            # Some incomplete input cannot be completed with a blank line.
            if shell.buffer:
                shell.errors.append("Incomplete Python input discarded at end of response.")
            return response
    finally:

        # Discard pending input, not the effects of already executed lines.
        shell.resetbuffer()
        if wrote_text and not at_line_start:
            print(flush=True)

        # Include local execution time, even on failure or interruption.
        duration = time.perf_counter() - started
        ttft = f"{first_token:.3f}s" if first_token is not None else "n/a"
        print(f"[Time to first token: {ttft} | Total duration: {duration:.3f}s]",
              file=sys.stderr, flush=True)
