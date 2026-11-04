"""Keep the CPU experiment's grading independent of model-visible context."""

import io
import sys
import json
import hashlib
import tempfile
import unittest
import subprocess
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from model.train.src.resources import exclusive_cpu
from dataset.benchmark.src.cpu_evaluate import trial, evaluate, completion
from dataset.benchmark.src.serialization import CODE, TASK, CONSOLE, END_CODE


class Worker:
    def __init__(self, fail_check=False):
        self.sources = []
        self.fail_check = fail_check

    def execute(self, source, action="execute"):
        self.sources.append((source, action))
        return {
            "stdout": "[4, 7]" if source == "print(values)" else "",
            "stderr": "",
            "exception": "AssertionError" if self.fail_check and source.startswith("assert") else None,
            "done": "ok" if "done(" in source else None,
        }


class CPUProofTests(unittest.TestCase):
    def test_completion_preserves_stop_order_and_end_marker_requirement(self):
        for stopping_word in (END_CODE, TASK, CONSOLE, CODE, ""):
            with self.subTest(stopping_word=stopping_word):
                response = io.StringIO(json.dumps({"content": "result = 11", "stopping_word": stopping_word}))
                with patch("dataset.benchmark.src.cpu_evaluate.urllib.request.urlopen", return_value=response) as urlopen:
                    self.assertEqual(completion("Sum values"), ("result = 11", stopping_word == END_CODE))
                payload = json.loads(urlopen.call_args.args[0].data)
                self.assertEqual(payload["stop"], [END_CODE, TASK, CONSOLE, CODE])
                self.assertEqual(payload["prompt"], "Sum values")

    def record(self):
        return {
            "id": "test",
            "family": "sum",
            "setup": "values = [4, 7]",
            "instruction": "Sum existing values into result.",
            "code": "REFERENCE_CODE_MUST_NOT_LEAK",
            "inspect": "REFERENCE_INSPECTION_MUST_NOT_LEAK",
            "check": "assert result == 11",
            "followup": None,
        }

    def test_setup_and_oracle_are_not_model_context(self):
        worker = Worker()
        prompts = []
        scripts = iter(["print(values)", "result = sum(values)\ndone(str(result))"])

        def complete(prompt):
            prompts.append(prompt)
            return next(scripts), True

        record = self.record()
        result = trial(worker, record, complete)
        self.assertEqual(result["status"], "passed")
        for prompt in prompts:
            for hidden in (record["setup"], record["check"], record["code"], record["inspect"]):
                self.assertNotIn(hidden, prompt)
        self.assertIn("[4, 7]", prompts[1])
        self.assertEqual(worker.sources[0], (record["setup"], "reset"))
        self.assertEqual(worker.sources[-1][0], record["check"])

    def test_done_claim_does_not_bypass_oracle(self):
        result = trial(Worker(fail_check=True), self.record(), lambda prompt: ("result = 999\ndone('success')", True))
        self.assertEqual(result["status"], "incorrect")

    def test_missing_end_marker_is_not_executed(self):
        worker = Worker()
        result = trial(worker, self.record(), lambda prompt: ("result = 11", False))
        self.assertEqual(result["status"], "boundary_error")
        self.assertEqual(len(worker.sources), 1)

    def test_followup_is_a_new_task_in_the_same_workspace(self):
        record = self.record()
        record["followup"] = {"instruction": "Append 9 in place.", "check": "assert result[-1] == 9"}
        prompts = []

        def complete(prompt):
            prompts.append(prompt)
            return "done('ok')", True

        worker = Worker()
        result = trial(worker, record, complete)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(sum(action == "reset" for _, action in worker.sources), 1)
        self.assertIn("Append 9 in place.", prompts[1])
        self.assertNotIn("assert result[-1]", prompts[1])

    def final_request(self, directory, *, final=False, limit=None):
        root = Path(directory)
        tasks = root / "test-tasks.jsonl"
        tasks.write_text("{}\n")
        manifest = {"splits": {"test": {"sha256": hashlib.sha256(tasks.read_bytes()).hexdigest()}}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return SimpleNamespace(data=root, split="test", final=final, limit=limit)

    def test_final_requires_explicit_authorization_before_loading_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.final_request(directory)
            with self.assertRaisesRegex(ValueError, "requires --final"):
                evaluate(args)

    def test_final_cannot_be_reduced_to_a_selected_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.final_request(directory, final=True, limit=10)
            with self.assertRaisesRegex(ValueError, "cannot be reduced"):
                evaluate(args)

    def test_changed_frozen_tasks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.final_request(directory, final=True)
            (args.data / "test-tasks.jsonl").write_text('{"modified":true}\n')
            with self.assertRaisesRegex(ValueError, "tasks have changed"):
                evaluate(args)

    def test_cpu_experiments_wait_for_each_other_without_polling(self):
        source = "from model.train.src.resources import exclusive_cpu\nwith exclusive_cpu(): print('acquired', flush=True)"
        with exclusive_cpu():
            child = subprocess.Popen([sys.executable, "-c", source], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertIn("Waiting for", child.stdout.readline())
                self.assertIsNone(child.poll())
            except BaseException:
                child.kill()
                child.communicate()
                raise
        output, error = child.communicate(timeout=10)
        self.assertEqual(child.returncode, 0, error)
        self.assertIn("acquired", output)
