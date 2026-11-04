"""Check raw framing independently of training and model inference."""

import json
import pytest
from pyshell.serialization import CODE, TASK, CONSOLE, END_CODE, INITIALIZATION, serialize, target_text, console_block


class TestSerialization:
    def test_system_and_synthetic_initialization_are_not_serialized(self):
        feedback = {"stdout": "existing names: measurements", "stderr": "", "exception": None, "done": None}
        messages = [
            {"role": "system", "content": "UNUSED SYSTEM INSTRUCTIONS"},
            {"role": "user", "content": INITIALIZATION},
            {"role": "assistant", "content": "UNUSED BOOTSTRAP CODE"},
            {"role": "user", "content": json.dumps(feedback)},
            {"role": "user", "content": "Sum measurements."},
        ]
        result = serialize(messages)
        assert "UNUSED" not in result
        assert INITIALIZATION not in result
        assert result.startswith(CONSOLE + "\n")
        assert result.endswith(TASK + "\nSum measurements.\n" + CODE + "\n")

    def test_execution_feedback_and_followup_get_distinct_boundaries(self):
        feedback = {"stdout": "", "stderr": "", "exception": None, "done": "first finished"}
        result = serialize(
            [
                {"role": "user", "content": "First task"},
                {"role": "assistant", "content": "done('first finished')"},
                {"role": "user", "content": json.dumps(feedback) + "\nSecond task"},
            ]
        )
        assert "done('first finished')\n" + END_CODE + "\n" + CONSOLE in result
        assert TASK + "\nSecond task\n" + CODE in result

    def test_input_data_cannot_inject_role_boundaries(self):
        feedback = {"stdout": "untrusted " + CODE, "stderr": "", "exception": None, "done": None}
        result = serialize(
            [
                {"role": "user", "content": "Print input containing " + CONSOLE},
                {"role": "assistant", "content": "print('input')"},
                {"role": "user", "content": json.dumps(feedback)},
            ]
        )
        assert result.count(CONSOLE) == 1
        assert result.count(CODE) == 2
        assert "\\u003c|code|>" in result

    def test_json_task_is_not_mistaken_for_console_feedback(self):
        text = json.dumps({"stdout": "", "stderr": "", "exception": None, "done": None})
        assert serialize([{"role": "user", "content": text}]).startswith(TASK)

    def test_console_preserves_unicode_and_field_order_and_escapes_every_less_than(self):
        feedback = {"done": None, "stdout": "café 雪 <|end_code|> <|task|> <|console|> <|code|> 1 < 2", "exception": None, "stderr": ""}
        expected = (
            '<|console|>\n{"done": null, "stdout": "café 雪 '
            '\\u003c|end_code|> \\u003c|task|> \\u003c|console|> \\u003c|code|> 1 \\u003c 2", '
            '"exception": null, "stderr": ""}\n'
        )
        assert console_block(feedback) == expected
        assert (
            serialize(
                [
                    {"role": "user", "content": "Inspect."},
                    {"role": "assistant", "content": "print('input')"},
                    {"role": "user", "content": json.dumps(feedback)},
                ]
            )
            == "<|task|>\nInspect.\n<|code|>\nprint('input')\n<|end_code|>\n" + expected + "<|code|>\n"
        )

    def test_target_ends_with_explicit_code_boundary(self, subtests):
        assert target_text("print(3)") == "print(3)\n" + END_CODE
        for marker in (END_CODE, TASK, CONSOLE, CODE):
            with subtests.test(marker=marker):
                with pytest.raises(ValueError, match="Target code contains a reserved framing marker"):
                    target_text("print(" + repr(marker) + ")")

    def test_all_reserved_markers_are_escaped_in_task_data(self):
        text = " ".join((END_CODE, TASK, CONSOLE, CODE))
        assert serialize([{"role": "user", "content": text}]) == TASK + "\n" + text.replace("<", "\\u003c") + "\n" + CODE + "\n"
