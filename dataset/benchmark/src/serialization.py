"""Raw task/code/console framing shared by training and inference."""

import json

TASK = "<|task|>"
CODE = "<|code|>"
CONSOLE = "<|console|>"
END_CODE = "<|end_code|>"
FORMAT = "task-code-console-v1"
INITIALIZATION = "Initialize the Python terminal, report the preloaded names, and display ui's docstring."
MARKERS = (END_CODE, TASK, CONSOLE, CODE)


def escape_text(text: str) -> str:
    """Prevent input data from impersonating framing markers."""
    for marker in MARKERS:
        text = text.replace(marker, marker.replace("<", "\\u003c"))
    return text


def feedback_and_task(text: str) -> tuple[dict | None, str]:
    """Split execution feedback from an optional follow-up instruction."""
    stripped = text.lstrip()
    try:
        value, offset = json.JSONDecoder().raw_decode(stripped)
    except ValueError:
        return None, text
    if isinstance(value, dict) and {"stdout", "stderr", "exception", "done"}.issubset(value):
        return value, stripped[offset:].strip()
    return None, text


def console_block(feedback: dict) -> str:
    encoded = json.dumps(feedback, ensure_ascii=False).replace("<", "\\u003c")
    return f"{CONSOLE}\n{encoded}\n"


def serialize(messages: list[dict[str, str]]) -> str:
    """Render context without a system role or a pretrained chat template."""
    turns = [message for message in messages if message["role"] != "system"]
    expects_console = False
    if len(turns) >= 3 and turns[0] == {"role": "user", "content": INITIALIZATION} and turns[1]["role"] == "assistant":
        turns = turns[2:]
        expects_console = True
    parts = []
    for turn in turns:
        content = turn["content"]
        if turn["role"] == "assistant":
            if any(marker in content for marker in MARKERS):
                raise ValueError("Code history contains a reserved framing marker")
            parts.append(f"{CODE}\n{content}\n{END_CODE}\n")
            expects_console = True
        elif turn["role"] == "user":
            feedback, task = feedback_and_task(content) if expects_console else (None, content)
            if feedback is not None:
                parts.append(console_block(feedback))
            if task.strip():
                parts.append(f"{TASK}\n{escape_text(task)}\n")
            expects_console = False
        else:
            raise ValueError(f"Unsupported source role: {turn['role']}")
    return "".join(parts) + CODE + "\n"


def target_text(code: str) -> str:
    """Supervise the response and its explicit stop boundary."""
    if any(marker in code for marker in MARKERS):
        raise ValueError("Target code contains a reserved framing marker")
    return code + "\n" + END_CODE
