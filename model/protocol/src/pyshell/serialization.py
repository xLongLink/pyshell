"""Raw task/code/console framing shared by training and inference."""

import json
from collections.abc import Mapping, Sequence

TASK = "<|task|>"
CODE = "<|code|>"
CONSOLE = "<|console|>"
END_CODE = "<|end_code|>"
FORMAT = "task-code-console-v1"
MARKERS = (END_CODE, TASK, CONSOLE, CODE)


def escape_text(text: str) -> str:
    """Prevent input data from impersonating framing markers."""
    for marker in MARKERS:
        text = text.replace(marker, marker.replace("<", "\\u003c"))
    return text


def console_block(feedback: dict) -> str:
    """Frame execution feedback without allowing injected markers."""
    encoded = json.dumps(feedback, ensure_ascii=False).replace("<", "\\u003c")
    return f"{CONSOLE}\n{encoded}\n"


def serialize(messages: Sequence[Mapping[str, str | dict]]) -> str:
    """Render context without a system role or a pretrained chat template."""
    parts = []
    for turn in messages:
        if turn["role"] == "system":
            continue
        content = turn["content"]
        if turn["role"] in {"assistant", "user"} and not isinstance(content, str):
            raise TypeError("Text history content must be a string")
        if turn["role"] == "assistant":
            if any(marker in content for marker in MARKERS):
                raise ValueError("Code history contains a reserved framing marker")
            parts.append(f"{CODE}\n{content}\n{END_CODE}\n")
        elif turn["role"] == "console":
            if not isinstance(content, dict):
                raise TypeError("Console content must be a feedback dictionary")
            parts.append(console_block(content))
        elif turn["role"] == "user":
            if content.strip():
                parts.append(f"{TASK}\n{escape_text(content)}\n")
        else:
            raise ValueError(f"Unsupported source role: {turn['role']}")
    return "".join(parts) + CODE + "\n"


def target_text(code: str) -> str:
    """Supervise the response and its explicit stop boundary."""
    if any(marker in code for marker in MARKERS):
        raise ValueError("Target code contains a reserved framing marker")
    return code + "\n" + END_CODE
