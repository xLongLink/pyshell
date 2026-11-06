import json
from collections.abc import Mapping

TASK = "<|task|>"
CODE = "<|code|>"
CONSOLE = "<|console|>"
END_CODE = "<|end_code|>"
MARKERS = (END_CODE, TASK, CONSOLE, CODE)


def _block(turn: Mapping[str, str | dict]) -> str:
    role = turn["role"]
    if role == "system":
        return ""
    content = turn["content"]
    if role in {"assistant", "user"} and not isinstance(content, str):
        raise TypeError("Text history content must be a string")
    if role == "assistant":
        if any(marker in content for marker in MARKERS):
            raise ValueError("Code history contains a reserved framing marker")
        return f"{CODE}\n{content}\n{END_CODE}\n"
    if role == "console":
        if not isinstance(content, dict):
            raise TypeError("Console content must be a feedback dictionary")
        encoded = json.dumps(content, ensure_ascii=False).replace("<", "\\u003c")
        return f"{CONSOLE}\n{encoded}\n"
    if role == "user":
        if not content.strip():
            return ""
        for marker in MARKERS:
            content = content.replace(marker, marker.replace("<", "\\u003c"))
        return f"{TASK}\n{content}\n"
    raise ValueError(f"Unsupported source role: {role}")


class Prompt:
    """Snapshot each appended message as a serialized block."""

    def __init__(self) -> None:
        self._blocks: list[str] = []

    def append(self, message: Mapping[str, str | dict]) -> None:
        """Validate and serialize a new message without revisiting history."""
        self._blocks.append(_block(message))

    def render(self) -> str:
        """Join the saved blocks with the next code boundary."""
        return "".join(self._blocks) + CODE + "\n"


def target_text(code: str) -> str:
    """Supervise the response and its explicit stop boundary."""
    if any(marker in code for marker in MARKERS):
        raise ValueError("Target code contains a reserved framing marker")
    return code + "\n" + END_CODE
