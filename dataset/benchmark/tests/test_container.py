"""Opt-in execution-backend integration without paid model requests."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from dataset.benchmark.src.cases import CASES
from pydantic_ai.models.function import FunctionModel
from dataset.benchmark.src.runner import Limits, run_trial
from dataset.benchmark.src.backend import ContainerBackend
from dataset.benchmark.src.recording import TokenBudget
from dataset.benchmark.tests.test_cases import SOLUTIONS
from dataset.benchmark.tests.test_harness import StoppedModel


@unittest.skipUnless(
    os.environ.get("BENCHMARK_CONTAINER_TESTS") == "1", "Set BENCHMARK_CONTAINER_TESTS=1 after building the benchmark image"
)
class ContainerTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_cases_through_real_execution_backend(self):
        with TemporaryDirectory(dir="/tmp/opencode") as temporary:
            directory = Path(temporary)
            backend = ContainerBackend(8020, directory / "backend.log")
            budget = TokenBudget(1_000_000)
            try:
                for case in CASES:
                    with self.subTest(case=case.id):
                        await backend.fresh()
                        solutions = iter(SOLUTIONS[case.id])

                        async def stream(_messages, _info, solutions=solutions):
                            yield next(solutions) + "\ndone('Verified reference execution')"

                        model = StoppedModel(FunctionModel(stream_function=stream))
                        result = await run_trial(
                            case,
                            repeat=1,
                            model=model,
                            shell_url=backend.url,
                            directory=directory / case.id,
                            limits=Limits(),
                            budget=budget,
                        )
                        self.assertEqual(result["status"], "passed", result["error"])
                        self.assertEqual(result["calls"], len(case.steps))
            finally:
                await backend.close()
