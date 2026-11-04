"""Check generated response boundaries without loading a model."""

import pytest
from pyshell.parsing import parse_completion
from pyshell.serialization import CODE, TASK, CONSOLE, END_CODE


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("code" + TASK + END_CODE, ("code", False)),
        ("code" + CONSOLE + END_CODE, ("code", False)),
        ("code" + CODE + END_CODE, ("code", False)),
        ("code" + END_CODE + CONSOLE, ("code", True)),
        ("code" + END_CODE + "ignored", ("code", True)),
        ("code", ("code", False)),
        ("", ("", False)),
        (END_CODE, ("", True)),
    ],
)
def test_completion_uses_the_first_boundary(text, expected):
    assert parse_completion(text) == expected
