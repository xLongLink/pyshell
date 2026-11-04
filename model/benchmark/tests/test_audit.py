"""Check execution grading and sampling without running model providers."""

import json
import httpx2
import pytest
from pathlib import Path
from argparse import Namespace
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch
from pyshell.serialization import CODE, TASK, CONSOLE, END_CODE
from model.benchmark.src.audit import rollout, evaluate, execute_candidate
from model.benchmark.src.sampling import balanced_sample
from model.benchmark.tests.test_harness import ExecutionTransport


class TestSampling:
    def test_sampling_is_balanced_deterministic_and_excludes_inspection(self):
        records = [{"id": f"{family}-{i}", "family": family, "inspect_only": i == 0} for family in ("a", "b", "c") for i in range(5)]
        selected = balanced_sample(records, 6)
        assert selected == balanced_sample(records, 6)
        assert not any(record["inspect_only"] for record in selected)
        assert {family: sum(record["family"] == family for record in selected) for family in ("a", "b", "c")} == {"a": 2, "b": 2, "c": 2}

    def test_insufficient_records_are_rejected(self):
        with pytest.raises(ValueError, match="Only 0 eligible examples available"):
            balanced_sample([], 1)


class Backend:
    url = "http://testserver"

    async def fresh(self):
        pass


class TestEvaluation:
    @pytest.mark.anyio
    async def test_empty_input_is_rejected_without_starting_backend_or_client(self):
        with TemporaryDirectory(dir="/tmp/opencode") as temporary:
            root = Path(temporary)
            args = Namespace(data=root / "data.jsonl", output=root / "audit", label="empty", max_rounds=4)
            args.data.write_text("")
            with (
                patch("model.benchmark.src.audit.ContainerBackend") as backend,
                patch("model.benchmark.src.audit.httpx2.AsyncClient") as client,
                pytest.raises(ValueError, match="^Execution audit requires at least one record$"),
            ):
                await evaluate(args)
            backend.assert_not_called()
            client.assert_not_called()
            assert not (args.output / "summary.json").exists()

    @pytest.mark.anyio
    async def test_mixed_results_preserve_summary_and_response_contracts(self):
        invalid_hf = {"source": "result = (", "end_marker_seen": True}
        valid_hf = {"source": "done('ok')", "end_marker_seen": True}
        unfinished_hf = {"source": "done('ok')", "end_marker_seen": False}
        records = [
            {"id": "one", "audit_split": "z", "family": "z", "hf_prediction": invalid_hf, "hf_target_loss": 0.8},
            {"id": "two", "audit_split": "a", "family": "z", "hf_prediction": valid_hf, "hf_target_loss": 0.4},
            {"id": "three", "audit_split": "z", "family": "a", "hf_prediction": valid_hf, "hf_target_loss": 0.2},
            {"id": "four", "audit_split": "a", "family": "a", "hf_prediction": unfinished_hf},
            {"id": "five", "audit_split": "z", "family": "z", "hf_target_loss": 0},
            {"id": "six", "audit_split": "a", "family": "z", "hf_prediction": None},
        ]
        rows = []
        for record, status, valid_steps in zip(
            records,
            ("passed", "timeout", "passed", "round_limit", "incorrect", "passed"),
            ([True], [], [True, True], [True, False], [True], [True]),
        ):
            record.update(prompt=record["id"], token_ids=[1])
            rows.append(
                {
                    "id": record["id"],
                    "split": record["audit_split"],
                    "family": record["family"],
                    "steps": [{"source": "pass" if valid else "(", "syntax_valid": valid} for valid in valid_steps],
                    "syntax_valid": bool(valid_steps) and all(valid_steps),
                    "status": status,
                    "error": "Rollout exceeded 90 seconds" if status == "timeout" else None,
                    "first_step_passed": status == "passed" and len(valid_steps) == 1,
                }
            )
        expected = {
            "label": "mixed",
            "scope": "up to 4 code/console turns from reference prior context, verified by execution",
            "examples": 6,
            "splits": {
                "a": {
                    "examples": 3,
                    "passed": 1,
                    "syntax_valid": 1,
                    "generated_scripts": 3,
                    "valid_scripts": 2,
                    "tokenization_matches_hf": 2,
                    "hf_target_losses": [0.4],
                    "by_family": {"a": {"examples": 1, "passed": 0}, "z": {"examples": 2, "passed": 1}},
                },
                "z": {
                    "examples": 3,
                    "passed": 2,
                    "syntax_valid": 3,
                    "generated_scripts": 4,
                    "valid_scripts": 4,
                    "tokenization_matches_hf": 2,
                    "hf_target_losses": [0.8, 0.2, 0],
                    "by_family": {"a": {"examples": 1, "passed": 1}, "z": {"examples": 2, "passed": 1}},
                },
            },
            "hf_comparison": {
                "examples": 4,
                "hf_passed": 1,
                "gguf_passed": 1,
                "scope": "first-step completion correctness; HF gets one generation, not a rollout",
            },
        }
        with TemporaryDirectory(dir="/tmp/opencode") as temporary:
            root = Path(temporary)
            args = Namespace(data=root / "data.jsonl", output=root / "audit", label="mixed", max_rounds=4)
            args.data.write_text("\n".join(json.dumps(record) for record in records))
            original_client = httpx2.AsyncClient

            def tokenize(request):
                assert request.url.path == "/tokenize"
                prompt = json.loads(request.content)["content"]
                return httpx2.Response(200, json={"tokens": [2] if prompt in ("two", "five") else [1]})

            def client(**kwargs):
                return original_client(**kwargs, transport=httpx2.MockTransport(tokenize))

            async def run_rollout(_backend, _client, record, max_rounds):
                assert max_rounds == 4
                index = records.index(record)
                if index:
                    previous = json.loads((args.output / "summary.json").read_text())
                    assert previous["examples"] == index
                    assert list(previous["splits"]) == sorted(previous["splits"])
                    for split in previous["splits"].values():
                        assert list(split["by_family"]) == sorted(split["by_family"])
                    responses = [json.loads(line) for line in (args.output / "responses.jsonl").read_text().splitlines()]
                    assert [row["id"] for row in responses] == [r["id"] for r in records[:index]]
                if record["id"] == "two":
                    raise TimeoutError
                return dict(rows[index])

            backend = AsyncMock()
            with (
                patch("model.benchmark.src.audit.ContainerBackend", return_value=backend),
                patch("model.benchmark.src.audit.httpx2.AsyncClient", side_effect=client),
                patch("model.benchmark.src.audit.rollout", side_effect=run_rollout),
                patch(
                    "model.benchmark.src.audit.execute_candidate",
                    side_effect=[TimeoutError(), {"status": "passed", "error": None, "done": "ok"}],
                ) as execute,
                patch("builtins.print") as output,
            ):
                assert await evaluate(args) == 0
            assert json.loads((args.output / "summary.json").read_text()) == expected
            assert json.loads(output.call_args.args[0]) == expected
            backend.close.assert_awaited_once()
            assert [call.args[1]["id"] for call in execute.await_args_list] == ["two", "three"]
            expected_rows = []
            for record, row in zip(records, rows):
                response = dict(row, tokenization_matches_hf=record["id"] not in ("two", "five"))
                if "hf_target_loss" in record:
                    response["hf_target_loss"] = record["hf_target_loss"]
                if record.get("hf_prediction") is not None:
                    response["hf_prediction"] = record["hf_prediction"]
                expected_rows.append(response)
            expected_rows[1]["hf_execution"] = {"status": "timeout", "error": "HF candidate execution exceeded 45 seconds"}
            expected_rows[2]["hf_execution"] = {"status": "passed", "error": None, "done": "ok"}
            responses = [json.loads(line) for line in (args.output / "responses.jsonl").read_text().splitlines()]
            assert responses == expected_rows

    @pytest.mark.anyio
    async def test_multistep_rollouts_write_each_summary_and_log_progress(self):
        transport = ExecutionTransport()
        original_client = httpx2.AsyncClient
        backend = AsyncMock()
        backend.url = Backend.url
        backend.fresh.side_effect = transport.reset
        scripts = iter(["print(values)", "result = sum(values)\ndone('ok')"] * 10)
        with TemporaryDirectory(dir="/tmp/opencode") as temporary:
            root = Path(temporary)
            args = Namespace(data=root / "data.jsonl", output=root / "audit", label="multistep", max_rounds=4)
            records = [
                {
                    "id": str(index),
                    "audit_split": "known",
                    "family": "sum",
                    "token_ids": [1],
                    "setup": "values = [1, 2, 3]",
                    "previous": [],
                    "check": "assert result == 6",
                    "prompt": "Sum values.\n" + CODE + "\n",
                }
                for index in range(10)
            ]
            args.data.write_text("\n".join(json.dumps(record) for record in records))

            def expected_summary(count):
                return {
                    "label": "multistep",
                    "scope": "up to 4 code/console turns from reference prior context, verified by execution",
                    "examples": count,
                    "splits": {
                        "known": {
                            "examples": count,
                            "passed": count,
                            "syntax_valid": count,
                            "generated_scripts": 2 * count,
                            "valid_scripts": 2 * count,
                            "tokenization_matches_hf": count,
                            "hf_target_losses": [],
                            "by_family": {"sum": {"examples": count, "passed": count}},
                        }
                    },
                    "hf_comparison": {
                        "examples": 0,
                        "hf_passed": 0,
                        "gguf_passed": 0,
                        "scope": "first-step completion correctness; HF gets one generation, not a rollout",
                    },
                }

            tokenizations = 0

            def generate(request):
                nonlocal tokenizations
                if request.url.path == "/tokenize":
                    if tokenizations:
                        summary = json.loads((args.output / "summary.json").read_text())
                        assert summary == expected_summary(tokenizations)
                        assert len((args.output / "responses.jsonl").read_text().splitlines()) == tokenizations
                    tokenizations += 1
                    return httpx2.Response(200, json={"tokens": [1]})
                assert request.url.path == "/completion"
                return httpx2.Response(200, json={"content": next(scripts), "stopping_word": END_CODE})

            def client(**kwargs):
                handler = generate if kwargs["base_url"] == "http://127.0.0.1:8030" else transport
                return original_client(**kwargs, transport=httpx2.MockTransport(handler))

            with (
                patch("model.benchmark.src.audit.ContainerBackend", return_value=backend),
                patch("model.benchmark.src.audit.httpx2.AsyncClient", side_effect=client),
                patch("builtins.print") as output,
            ):
                assert await evaluate(args) == 0
            assert json.loads((args.output / "summary.json").read_text()) == expected_summary(10)
            output.assert_any_call("Execution audit 10/10: 10 passed", flush=True)
            assert json.loads(output.call_args.args[0]) == expected_summary(10)
            responses = [json.loads(line) for line in (args.output / "responses.jsonl").read_text().splitlines()]
            assert all(row["status"] == "passed" and (not row["first_step_passed"]) for row in responses)
            assert all(len(row["steps"]) == 2 for row in responses)
            backend.close.assert_awaited_once()


class TestExecutionAudit:
    async def grade(self, code, previous=None):
        transport = ExecutionTransport()
        original_client = httpx2.AsyncClient

        def client(**kwargs):
            return original_client(**kwargs, transport=httpx2.MockTransport(transport))

        record = {"setup": "values = [1, 2, 3]", "previous": previous or [], "check": "assert result == sum(values)"}
        with patch("model.benchmark.src.audit.httpx2.AsyncClient", side_effect=client):
            return await execute_candidate(Backend(), record, code)

    @pytest.mark.anyio
    async def test_success_is_verified_by_state(self):
        assert (await self.grade("result = sum(values)\ndone('ok')"))["status"] == "passed"
        assert (await self.grade("result = 999\ndone('success')"))["status"] == "incorrect"

    @pytest.mark.anyio
    async def test_completion_is_required(self):
        assert (await self.grade("result = sum(values)"))["status"] == "not_completed"

    @pytest.mark.anyio
    async def test_previous_partial_updates_survive_errors(self):
        previous = ["values.append(4)\nraise ValueError('after successful append')"]
        assert (await self.grade("result = sum(values)\ndone('ok')", previous))["status"] == "passed"

    @pytest.mark.anyio
    async def test_fixture_failures_stop_execution_after_setup(self, subtests):
        original_client = httpx2.AsyncClient
        for operation in ("candidate", "rollout"):
            for bootstrap_fails, setup_fails in ((True, False), (False, True), (True, True)):
                with subtests.test(operation=operation, bootstrap_fails=bootstrap_fails, setup_fails=setup_fails):
                    transport = ExecutionTransport()
                    generator = AsyncMock()
                    bootstrap = "stage = 'bootstrap'"
                    if bootstrap_fails:
                        bootstrap += "\nraise ValueError('bootstrap failed')"
                    setup = "assert stage == 'bootstrap'\nstage = 'setup'"
                    if setup_fails:
                        setup += "\nraise ValueError('setup failed')"
                    record = {
                        "id": "test",
                        "audit_split": "known",
                        "family": "sum",
                        "setup": setup,
                        "previous": ["stage = 'previous'"],
                        "check": "stage = 'check'",
                        "prompt": "Complete the task.\n" + CODE + "\n",
                    }

                    def client(**kwargs):
                        return original_client(**kwargs, transport=httpx2.MockTransport(transport))

                    error = "bootstrap failed" if bootstrap_fails else "setup failed"
                    candidate = execute_candidate if operation == "candidate" else rollout
                    arguments = (
                        (Backend(), record, "stage = 'candidate'\ndone('ok')")
                        if operation == "candidate"
                        else (Backend(), generator, record, 4)
                    )
                    with (
                        patch("model.benchmark.src.audit.httpx2.AsyncClient", side_effect=client),
                        patch("model.benchmark.src.audit.STARTUP_LINES", [bootstrap]),
                        pytest.raises(RuntimeError, match=f"^Fixture failed: ValueError: {error}$"),
                    ):
                        await candidate(*arguments)
                    assert transport.submitted == [bootstrap, setup]
                    assert transport.namespace["stage"] == "setup"
                    generator.post.assert_not_awaited()

    @pytest.mark.anyio
    async def test_rollout_allows_inspection_without_leaking_oracle(self):
        transport = ExecutionTransport()
        original_client = httpx2.AsyncClient
        payloads = []
        inspection = "print(values)\nprint(inspection_text)"
        scripts = iter([inspection, "result = sum(values)\ndone('ok')"])

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
            "setup": "values = [1, 2, 3]\ninspection_text = 'café 雪 <|end_code|> <|task|> <|console|> <|code|> 1 < 2'",
            "previous": [],
            "check": "assert result == 6",
            "prompt": "Sum values.\n" + CODE + "\n",
        }
        with patch("model.benchmark.src.audit.httpx2.AsyncClient", side_effect=client):
            result = await rollout(Backend(), Generator(), record, 4)
        assert result["status"] == "passed"
        assert len(result["steps"]) == 2
        for payload in payloads:
            assert payload["stop"] == [END_CODE, TASK, CONSOLE, CODE]
        assert not result["first_step_passed"]
        assert payloads[0]["prompt"] == record["prompt"]
        assert (
            payloads[1]["prompt"]
            == record["prompt"]
            + inspection
            + '\n<|end_code|>\n<|console|>\n{"stdout": "[1, 2, 3]\\ncafé 雪 \\u003c|end_code|> \\u003c|task|> \\u003c|console|> \\u003c|code|> 1 \\u003c 2\\n", "stderr": "", "exception": null, "done": null, "more_input": false}\n<|code|>\n'
        )
        assert record["check"] not in payloads[1]["prompt"]
