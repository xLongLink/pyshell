"""Interpret generated code and its protocol boundary."""

from pyshell.serialization import MARKERS, END_CODE


def parse_completion(text: str) -> tuple[str, bool]:
    """Return code before the first marker and whether it ended explicitly."""
    boundaries = [(text.index(marker), marker) for marker in MARKERS if marker in text]
    if not boundaries:
        return text, False
    offset, marker = min(boundaries)
    return text[:offset], marker == END_CODE
