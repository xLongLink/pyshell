"""Check balanced inspection and recovery supervision without a model."""

import io
import json
import pytest
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import Mock, patch
from importlib.util import find_spec


@pytest.mark.skipif(not (find_spec("pandas")), reason="Install the benchmark data extra")
class TestExecutedCurriculum:
    def prepare_records(self, records):
        from dataset.generation.src import executed
        from dataset.benchmark.src.reference import ReferenceWorkspace

        workspace = ReferenceWorkspace()
        worker = Mock(execute=Mock(wraps=workspace.execute))
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            (evaluation / "manifest.json").write_text(
                json.dumps({"splits": {"dev": {"template_ids": []}, "test": {"template_ids": [], "sha256": "synthetic-test"}}})
            )
            validation = b'{"id": "synthetic-validation"}\n'
            (evaluation / "validation.jsonl").write_bytes(validation)
            output = root / "output"
            with (
                patch.object(executed, "Worker", return_value=worker),
                patch.object(executed, "training_tasks", return_value=iter(records)),
                redirect_stdout(io.StringIO()),
            ):
                executed.prepare(output, evaluation)
            rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
            assert (output / "validation.jsonl").read_bytes() == validation
        worker.close.assert_called_once_with()
        assert [call.args[0] for call in worker.execute.call_args_list if call.args[1:] == ("reset",)] == [
            record["setup"] for record in records for _ in range(2 if record["recovery_code"] else 1)
        ]
        return rows, workspace

    def test_training_template_cycle_wraps_after_six_pairs_for_every_family(self, subtests):
        from dataset.generation.src.executed import training_tasks
        from dataset.benchmark.src.task_types import FAMILIES

        records = list(training_tasks(8, 917341))
        assert len(records) == len(FAMILIES) * 16
        for family in FAMILIES:
            for variant in (0, 1):
                with subtests.test(family=family, variant=variant):
                    assert [record["template_id"] for record in records if record["family"] == family and record["variant"] == variant] == [
                        f"{family}-t{slot}" for slot in (0, 1, 2, 3, 4, 5, 0, 1)
                    ]

    def test_every_family_and_variant_has_supervised_inspection_and_a_verified_answer(self, subtests):
        from dataset.generation.src.executed import training_tasks
        from dataset.benchmark.src.task_types import FAMILIES

        records = list(training_tasks(2, 917341))
        assert len(records) == len(FAMILIES) * 4
        produced, _ = self.prepare_records(records)
        for record in records:
            with subtests.test(task=record["id"]):
                rows = [row for row in produced if row["id"].startswith(record["id"] + "-direct-")]
                assert rows[0]["completion"] == record["inspect"]
                assert record["code"] in [row["completion"] for row in rows]
                assert record["template_id"].rsplit("t", 1)[1] in {"0", "1"}
                assert [row["id"] for row in rows] == [
                    f"{record['id']}-direct-s{step}" for step in ([1, 3, 6] if record["followup"] else [1, 3])
                ]
                for row in rows:
                    assert record["check"] not in row["prompt"]

    def test_sum_and_mean_both_receive_direct_and_error_recovery_trajectories(self):
        from dataset.generation.src.executed import training_tasks

        records = [record for record in training_tasks(1, 917341) if record["family"] == "recover_execution"]
        assert {record["variant"] for record in records} == {0, 1}
        produced, _ = self.prepare_records(records)
        assert [row["id"] for row in produced] == [
            f"{record['id']}-{suffix}-s{step}"
            for record in records
            for suffix, steps in (("direct", (1, 3)), ("retry", (1, 3, 5)))
            for step in steps
        ]
        for record in records:
            direct = [row for row in produced if row["id"].startswith(record["id"] + "-direct-")]
            retry = [row for row in produced if row["id"].startswith(record["id"] + "-retry-")]
            assert len(direct) == 2
            assert len(retry) == 3
            assert "RuntimeError" in retry[-1]["prompt"]
            assert ".recover()" in direct[-1]["completion"]
            assert ".recover()" in retry[-1]["completion"]
            for suffix, rows in (("direct", direct), ("retry", retry)):
                assert all(row["id"].startswith(f"{record['id']}-{suffix}-s") for row in rows)

    def test_followup_preserves_result_identity_and_hides_both_checks(self):
        from dataset.generation.src.executed import training_tasks

        for record in training_tasks(1, 917341):
            if record["followup"] is None:
                continue
            rows, workspace = self.prepare_records([record])
            assert workspace.namespace["result"] is workspace.namespace["saved_result"]
            assert record["followup"]["instruction"] in rows[-1]["prompt"]
            for row in rows:
                assert record["check"] not in row["prompt"]
                assert record["followup"]["check"] not in row["prompt"]
