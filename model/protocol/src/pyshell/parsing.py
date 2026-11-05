import re

from pyshell.serialization import MARKERS, END_CODE

BOUNDARY = re.compile("|".join(re.escape(marker) for marker in MARKERS))


def parse_completion(text: str) -> tuple[str, bool]:
    """Return code before the first marker and whether it ended explicitly."""
    match = BOUNDARY.search(text)
    if match is None:
        return text, False
    return text[:match.start()], match.group() == END_CODE
