"""Opt-in execution-backend integration without paid model requests."""

import os
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from model.benchmark.src.cases import CASES
from model.benchmark.src.runner import Limits, run_trial
from model.benchmark.src.backend import ContainerBackend
from pydantic_ai.models.function import FunctionModel
from model.benchmark.src.recording import TokenBudget
from model.benchmark.tests.test_cases import SOLUTIONS
from model.benchmark.tests.test_harness import StoppedModel


@pytest.mark.skipif(
    not (os.environ.get("BENCHMARK_CONTAINER_TESTS") == "1"), reason="Set BENCHMARK_CONTAINER_TESTS=1 after building the benchmark image"
)
class TestContainer:
    @pytest.mark.anyio
    async def test_all_cases_through_real_execution_backend(self, subtests):
        with TemporaryDirectory(dir="/tmp/opencode") as temporary:
            directory = Path(temporary)
            backend = ContainerBackend(8020, directory / "backend.log")
            budget = TokenBudget(1_000_000)
            try:
                for case in CASES:
                    with subtests.test(case=case.id):
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
                        assert result["status"] == "passed", result["error"]
                        assert result["calls"] == len(case.steps)
            finally:
                await backend.close()
