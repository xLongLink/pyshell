"""Check raw framing independently of training and model inference."""

import json
import unittest
from dataset.benchmark.src.serialization import CODE, TASK, CONSOLE, END_CODE, INITIALIZATION, serialize, target_text


class SerializationTests(unittest.TestCase):
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
        self.assertNotIn("UNUSED", result)
        self.assertNotIn(INITIALIZATION, result)
        self.assertTrue(result.startswith(CONSOLE + "\n"))
        self.assertTrue(result.endswith(TASK + "\nSum measurements.\n" + CODE + "\n"))

    def test_execution_feedback_and_followup_get_distinct_boundaries(self):
        feedback = {"stdout": "", "stderr": "", "exception": None, "done": "first finished"}
        result = serialize(
            [
                {"role": "user", "content": "First task"},
                {"role": "assistant", "content": "done('first finished')"},
                {"role": "user", "content": json.dumps(feedback) + "\nSecond task"},
            ]
        )
        self.assertIn("done('first finished')\n" + END_CODE + "\n" + CONSOLE, result)
        self.assertIn(TASK + "\nSecond task\n" + CODE, result)

    def test_input_data_cannot_inject_role_boundaries(self):
        feedback = {"stdout": "untrusted " + CODE, "stderr": "", "exception": None, "done": None}
        result = serialize(
            [
                {"role": "user", "content": "Print input containing " + CONSOLE},
                {"role": "assistant", "content": "print('input')"},
                {"role": "user", "content": json.dumps(feedback)},
            ]
        )
        self.assertEqual(result.count(CONSOLE), 1)
        self.assertEqual(result.count(CODE), 2)
        self.assertIn("\\u003c|code|>", result)

    def test_json_task_is_not_mistaken_for_console_feedback(self):
        text = json.dumps({"stdout": "", "stderr": "", "exception": None, "done": None})
        self.assertTrue(serialize([{"role": "user", "content": text}]).startswith(TASK))

    def test_target_ends_with_explicit_code_boundary(self):
        self.assertEqual(target_text("print(3)"), "print(3)\n" + END_CODE)
        with self.assertRaises(ValueError):
            target_text("print(" + repr(END_CODE) + ")")
