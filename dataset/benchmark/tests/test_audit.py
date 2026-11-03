"""Check execution grading and sampling without running model providers."""

import httpx2
import unittest
from unittest.mock import patch
from dataset.benchmark.src.audit import rollout, execute_candidate
from dataset.benchmark.src.sampling import balanced_sample
from dataset.benchmark.src.serialization import CODE, END_CODE
from dataset.benchmark.tests.test_harness import ExecutionTransport


class SamplingTests(unittest.TestCase):
    def test_sampling_is_balanced_deterministic_and_excludes_inspection(self):
        records = [{"id": f"{family}-{i}", "family": family, "inspect_only": i == 0} for family in ("a", "b", "c") for i in range(5)]
        selected = balanced_sample(records, 6)
        self.assertEqual(selected, balanced_sample(records, 6))
        self.assertFalse(any(record["inspect_only"] for record in selected))
        self.assertEqual(
            {family: sum(record["family"] == family for record in selected) for family in ("a", "b", "c")}, {"a": 2, "b": 2, "c": 2}
        )

    def test_insufficient_records_are_rejected(self):
        with self.assertRaises(ValueError):
            balanced_sample([], 1)


class Backend:
    url = "http://testserver"

    async def fresh(self):
        pass


class ExecutionAuditTests(unittest.IsolatedAsyncioTestCase):
    async def grade(self, code, previous=None):
        transport = ExecutionTransport()
        original_client = httpx2.AsyncClient

        def client(**kwargs):
            return original_client(**kwargs, transport=httpx2.MockTransport(transport))

        record = {"setup": "values = [1, 2, 3]", "previous": previous or [], "check": "assert result == sum(values)"}
        with patch("dataset.benchmark.src.audit.httpx2.AsyncClient", side_effect=client):
            return await execute_candidate(Backend(), record, code)

    async def test_success_is_verified_by_state(self):
        self.assertEqual((await self.grade("result = sum(values)\ndone('ok')"))["status"], "passed")
        self.assertEqual((await self.grade("result = 999\ndone('success')"))["status"], "incorrect")

    async def test_completion_is_required(self):
        self.assertEqual((await self.grade("result = sum(values)"))["status"], "not_completed")

    async def test_previous_partial_updates_survive_errors(self):
        previous = ["values.append(4)\nraise ValueError('after successful append')"]
        self.assertEqual((await self.grade("result = sum(values)\ndone('ok')", previous))["status"], "passed")

    async def test_rollout_allows_inspection_without_leaking_oracle(self):
        transport = ExecutionTransport()
        original_client = httpx2.AsyncClient
        payloads = []
        scripts = iter(["print(values)", "result = sum(values)\ndone('ok')"])

        class Generator:
            async def post(self, _url, *, json):
                payloads.append(json)
                return httpx2.Response(
                    200,
                    json={"content": next(scripts), "stopping_word": END_CODE},
                    request=httpx2.Request("POST", "http://testserver/completion"),
                )

        def client(**kwargs):
            return original_client(**kwargs, transport=httpx2.MockTransport(transport))

        record = {
            "id": "test",
            "audit_split": "known",
            "family": "sum",
            "setup": "values = [1, 2, 3]",
            "previous": [],
            "check": "assert result == 6",
            "prompt": "Sum values.\n" + CODE + "\n",
        }
        with patch("dataset.benchmark.src.audit.httpx2.AsyncClient", side_effect=client):
            result = await rollout(Backend(), Generator(), record, 4)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["steps"]), 2)
        self.assertFalse(result["first_step_passed"])
        self.assertIn("[1, 2, 3]", payloads[1]["prompt"])
        self.assertNotIn(record["check"], payloads[1]["prompt"])
