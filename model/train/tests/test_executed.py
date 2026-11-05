import io
import json
import stat
import tempfile
from collections import Counter
from contextlib import redirect_stdout
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from model.benchmark.src.corpus import read_jsonl, sha256


class FlushedOutput(io.StringIO):
    def flush(self):
        self.flushed = self.getvalue()
        super().flush()


@pytest.mark.skipif(
    not (find_spec("pandas")), reason="Install the benchmark data extra"
)
class TestExecutedCurriculum:
    def prepare_records(self, records):
        from model.benchmark.src.reference import ReferenceWorkspace
        from model.train.src.generation import executed

        workspace = ReferenceWorkspace()
        worker = Mock(execute=Mock(wraps=workspace.execute))
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            (evaluation / "manifest.json").write_text(
                json.dumps(
                    {
                        "splits": {
                            "dev": {"template_ids": []},
                            "test": {"template_ids": [], "sha256": "synthetic-test"},
                        }
                    }
                )
            )
            validation = b'{ "text": "caf\xc3\xa9" }\r\n\n'
            (evaluation / "validation.jsonl").write_bytes(validation)
            output = root / "output"
            with (
                patch.object(executed, "Worker", return_value=worker),
                patch.object(executed, "training_tasks", return_value=iter(records)),
                redirect_stdout(FlushedOutput()) as stdout,
            ):
                assert executed.prepare(output, evaluation) is None
            rows = [
                json.loads(line)
                for line in (output / "train.jsonl").read_text().splitlines()
            ]
            assert (output / "validation.jsonl").read_bytes() == validation
            assert read_jsonl(output / "train-tasks.jsonl") == [
                {key: value for key, value in record.items() if key != "expected"}
                for record in records
            ]
            manifest = {
                "experiment": "executed-transfer-curriculum-v1",
                "training_only": True,
                "seed": 917341,
                "pairs_per_family": 8,
                "training_records": len(rows),
                "tasks_by_family": dict(Counter(record["family"] for record in records)),
                "supervision_by_family": dict(Counter(row["family"] for row in rows)),
                "template_ids": sorted({record["template_id"] for record in records}),
                "feedback": "real network-isolated PythonShell execution",
                "recovery": "both sum and mean receive successful try/except and failed-read/recover/retry trajectories",
                "inspection": "supervised for every task",
                "evaluation_test_sha256": "synthetic-test",
                "splits": {
                    "train": {
                        "tasks": len(records),
                        "sha256": sha256(output / "train-tasks.jsonl"),
                    }
                },
                "training_records_sha256": sha256(output / "train.jsonl"),
                "validation_records_sha256": sha256(output / "validation.jsonl"),
                "generator_sha256": sha256(Path(executed.__file__)),
            }
            expected = (json.dumps(manifest, indent=2) + "\n").encode()
            assert (output / "manifest.json").read_bytes() == expected
            assert stdout.getvalue().encode() == expected
            assert stdout.flushed.encode() == expected
            assert all(
                stat.S_IMODE(path.stat().st_mode) == 0o444
                for path in output.iterdir()
            )
        worker.close.assert_called_once_with()
        assert [
            call.args[0]
            for call in worker.execute.call_args_list
            if call.args[1:] == ("reset",)
        ] == [
            record["setup"]
            for record in records
            for _ in range(2 if record["recovery_code"] else 1)
        ]
        return rows, workspace

    @pytest.mark.parametrize("split", ["dev", "test"])
    def test_held_out_templates_fail_before_publication(self, tmp_path, split):
        from model.benchmark.src.reference import ReferenceWorkspace
        from model.train.src.generation import executed

        records = list(executed.training_tasks(1, 917341))[:1]
        evaluation = tmp_path / "evaluation"
        evaluation.mkdir()
        frozen = {"splits": {name: {"template_ids": []} for name in ("dev", "test")}}
        frozen["splits"][split]["template_ids"] = [records[0]["template_id"]]
        (evaluation / "manifest.json").write_text(json.dumps(frozen))
        workspace = ReferenceWorkspace()
        worker = Mock(execute=workspace.execute)
        output = tmp_path / "output"
        with (
            patch.object(executed, "Worker", return_value=worker),
            patch.object(executed, "training_tasks", return_value=iter(records)),
            pytest.raises(ValueError, match="Training templates overlap"),
        ):
            executed.prepare(output, evaluation)
        assert list(output.iterdir()) == []
        worker.close.assert_called_once_with()

    def test_training_template_cycle_wraps_after_six_pairs_for_every_family(
        self, subtests
    ):
        from model.benchmark.src.transfer_families import FAMILIES
        from model.train.src.generation.executed import training_tasks

        records = list(training_tasks(8, 917341))
        assert len(records) == len(FAMILIES) * 16
        for family in FAMILIES:
            for variant in (0, 1):
                with subtests.test(family=family, variant=variant):
                    assert [
                        record["template_id"]
                        for record in records
                        if record["family"] == family and record["variant"] == variant
                    ] == [f"{family}-t{slot}" for slot in (0, 1, 2, 3, 4, 5, 0, 1)]

    def test_every_family_and_variant_has_supervised_inspection_and_a_verified_answer(
        self, subtests
    ):
        from model.benchmark.src.transfer_families import FAMILIES
        from model.train.src.generation.executed import training_tasks

        records = list(training_tasks(2, 917341))
        assert len(records) == len(FAMILIES) * 4
        produced, _ = self.prepare_records(records)
        for record in records:
            with subtests.test(task=record["id"]):
                rows = [
                    row
                    for row in produced
                    if row["id"].startswith(record["id"] + "-direct-")
                ]
                assert rows[0]["completion"] == record["inspect"]
                assert record["code"] in [row["completion"] for row in rows]
                assert record["template_id"].rsplit("t", 1)[1] in {"0", "1"}
                assert [row["id"] for row in rows] == [
                    f"{record['id']}-direct-s{step}"
                    for step in ([1, 3, 6] if record["followup"] else [1, 3])
                ]
                for row in rows:
                    assert record["check"] not in row["prompt"]

    def test_sum_and_mean_both_receive_direct_and_error_recovery_trajectories(self):
        from model.train.src.generation.executed import training_tasks

        records = [
            record
            for record in training_tasks(1, 917341)
            if record["family"] == "recover_execution"
        ]
        assert {record["variant"] for record in records} == {0, 1}
        produced, _ = self.prepare_records(records)
        assert [row["id"] for row in produced] == [
            f"{record['id']}-{suffix}-s{step}"
            for record in records
            for suffix, steps in (("direct", (1, 3)), ("retry", (1, 3, 5)))
            for step in steps
        ]
        for record in records:
            direct = [
                row
                for row in produced
                if row["id"].startswith(record["id"] + "-direct-")
            ]
            retry = [
                row
                for row in produced
                if row["id"].startswith(record["id"] + "-retry-")
            ]
            assert len(direct) == 2
            assert len(retry) == 3
            assert "RuntimeError" in retry[-1]["prompt"]
            assert ".recover()" in direct[-1]["completion"]
            assert ".recover()" in retry[-1]["completion"]
            for suffix, rows in (("direct", direct), ("retry", retry)):
                assert all(
                    row["id"].startswith(f"{record['id']}-{suffix}-s") for row in rows
                )

    def test_followup_preserves_result_identity_and_hides_both_checks(self):
        from model.train.src.generation.executed import training_tasks

        for record in training_tasks(1, 917341):
            if record["followup"] is None:
                continue
            rows, workspace = self.prepare_records([record])
            assert workspace.namespace["result"] is workspace.namespace["saved_result"]
            assert record["followup"]["instruction"] in rows[-1]["prompt"]
            for row in rows:
                assert record["check"] not in row["prompt"]
                assert record["followup"]["check"] not in row["prompt"]
