import hashlib
import io
import json
import statistics
import tempfile
from contextlib import redirect_stdout
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest

AVAILABLE = find_spec("pandas") is not None


@pytest.mark.skipif(
    not (AVAILABLE), reason="Install the benchmark project's data extra"
)
class TestTransfer:
    def test_task_names_preserve_random_data_and_metadata(self, subtests):
        from model.benchmark.src.transfer_families import FAMILIES
        from model.train.src.generation import transfer

        for family in FAMILIES:
            for index, seed in ((0, 183731), (1, 813797), (5, 731971)):
                with subtests.test(family=family, index=index, seed=seed):
                    with patch.object(transfer, "NAMES", ["original_input"]):
                        original = transfer.make_task(
                            family, index, 0, seed, "train", 0
                        )
                    with patch.object(transfer, "NAMES", ["chosen_input"]):
                        renamed = transfer.make_task(family, index, 0, seed, "train", 0)
                    assert renamed.extra == original.extra
                    assert renamed == json.loads(
                        json.dumps(original).replace(original.name, renamed.name)
                    )

    def test_transfer_records_preserve_frozen_output_and_execute_followups(
        self, subtests
    ):
        from model.benchmark.src.reference import ReferenceSession
        from model.benchmark.src.transfer_families import FAMILIES
        from model.train.src.generation.transfer import TEMPLATE_PARTITIONS, make_task

        records = []
        for family in FAMILIES:
            for split, slots in TEMPLATE_PARTITIONS.items():
                for slot in slots:
                    for pair in (0, 1):
                        for variant in (0, 1):
                            records.append(
                                make_task(family, pair, variant, 183731, split, slot)
                            )
        assert hashlib.sha256(json.dumps(records).encode()).hexdigest() == (
            "00bc2391482645a813775b451f8243620689c7589a4dd6daa6509fb091345d97"
        )
        for record in records:
            with subtests.test(record=record["id"], template=record["template_id"]):
                session = ReferenceSession(record)
                session.add(record["inspect"])
                session.add(record["code"])
                session.check(record["check"])
                if record["family"] == "stateful_followup":
                    assert set(record["followup"]) == {"instruction", "code", "check"}
                    session.followup(record["followup"], "followup")
                else:
                    assert record["followup"] is None

    def test_contrast_pair_retries_with_matching_seeds_in_variant_order(self, subtests):
        from model.train.src.generation.transfer import make_contrast_pair

        for successful_attempt in (0, 1, 31):
            with subtests.test(successful_attempt=successful_attempt):
                rejected = [{"expected": 0}, {"expected": 0}]
                accepted = [{"expected": 1}, {"expected": 2}]
                with patch(
                    "model.train.src.generation.transfer.make_task",
                    side_effect=rejected * successful_attempt + accepted,
                ) as make_task:
                    pair = make_contrast_pair("statistics", 3, 183731, "dev", 7)
                assert pair == accepted
                assert pair[0] is accepted[0]
                assert pair[1] is accepted[1]
                assert make_task.call_args_list == [
                    call(
                        "statistics",
                        3,
                        variant,
                        183731 + attempt * 1000000007,
                        "dev",
                        7,
                    )
                    for attempt in range(successful_attempt + 1)
                    for variant in (0, 1)
                ]

    def test_contrast_pair_raises_after_exhausting_retries(self):
        from model.train.src.generation.transfer import make_contrast_pair

        with patch(
            "model.train.src.generation.transfer.make_task",
            return_value={"expected": 0},
        ) as make_task:
            with pytest.raises(
                ValueError,
                match="^Could not construct a nontrivial contrast for statistics$",
            ):
                make_contrast_pair("statistics", 3, 183731, "dev", 7)
        assert make_task.call_args_list == [
            call("statistics", 3, variant, 183731 + attempt * 1000000007, "dev", 7)
            for attempt in range(32)
            for variant in (0, 1)
        ]

    def test_pairs_require_different_answers_and_cross_answers_fail(self, subtests):
        import numpy as np
        import pandas as pd

        from model.benchmark.src.reference import feedback
        from model.benchmark.src.transfer_families import FAMILIES
        from model.train.src.generation.transfer import make_contrast_pair

        for family in FAMILIES:
            with subtests.test(family=family):
                pair = make_contrast_pair(family, 0, 183731, "train", 1)
                assert pair[0]["expected"] != pair[1]["expected"]
                if family != "missing_input":
                    assert pair[0]["setup"] == pair[1]["setup"]
                    assert pair[0]["instruction"] != pair[1]["instruction"]
                for index, record in enumerate(pair):
                    namespace = {
                        "np": np,
                        "pd": pd,
                        "statistics": statistics,
                        "json": json,
                    }
                    exec(record["setup"], namespace)
                    assert feedback(record["code"], namespace)["exception"] is None
                    exec(record["check"], namespace)
                    with pytest.raises(AssertionError):
                        exec(pair[1 - index]["check"], namespace)

    def test_recovery_supervision_includes_a_real_failed_read_and_successful_retry(
        self,
    ):
        from model.train.src.generation.transfer import make_task, supervision

        record = make_task("recover_execution", 1, 1, 183731, "train", 2)
        record["inspect"] = "print('Inspect the existing reader without help().')"
        rows = supervision(record, include_inspection=True)
        assert len(rows) == 3
        assert "RuntimeError" in rows[-1]["prompt"]
        assert ".recover()" in rows[-1]["completion"]
        assert ".read()" in rows[-1]["completion"]

    def test_disabling_capture_preserves_execution_without_turns_for_every_family(
        self, subtests
    ):
        from model.benchmark.src.reference import (
            ReferenceSession,
            ReferenceWorkspace,
        )
        from pyshell.serialization import Prompt
        from model.benchmark.src.transfer_families import FAMILIES
        from model.train.src.generation.transfer import make_task, supervision

        sessions = []

        def start_session(*args, **kwargs):
            session = ReferenceSession(*args, **kwargs)
            sessions.append(session)
            return session

        for family in FAMILIES:
            for variant in (0, 1):
                for include_inspection in (False, True):
                    with subtests.test(
                        family=family, variant=variant, inspection=include_inspection
                    ):
                        record = make_task(family, 1, variant, 183731, "test", 8)
                        workspaces = [ReferenceWorkspace(), ReferenceWorkspace()]
                        calls = []
                        for index, workspace in enumerate(workspaces):
                            worker = Mock(execute=Mock(wraps=workspace.execute))
                            options = {} if index == 0 else {"capture_rows": False}
                            with (
                                patch(
                                    "model.train.src.generation.transfer.ReferenceSession",
                                    side_effect=start_session,
                                ),
                                patch(
                                    "model.benchmark.src.reference.Prompt.render",
                                    autospec=True,
                                    side_effect=Prompt.render,
                                ) as capture,
                            ):
                                rows = supervision(
                                    record,
                                    include_inspection,
                                    workspace=worker,
                                    **options,
                                )
                            if index == 0:
                                assert rows
                                assert capture.call_count == len(rows)
                            else:
                                assert rows == []
                                capture.assert_not_called()
                            calls.append(worker.execute.call_args_list)
                        assert calls[0] == calls[1]
                        assert sessions[-2].turns
                        assert sessions[-1].turns == []
                        if record["recovery_code"] and variant:
                            assert any(
                                turn["role"] == "console"
                                and "RuntimeError"
                                in (turn["content"]["exception"] or "")
                                for turn in sessions[-2].turns
                            )
                        if record["followup"]:
                            for workspace in workspaces:
                                assert (
                                    workspace.namespace["result"]
                                    is workspace.namespace["saved_result"]
                                )

    def test_disabling_capture_still_enforces_errors_and_oracle_checks(self, subtests):
        from model.train.src.generation.transfer import make_task, supervision

        for family, field in (
            ("statistics", "code"),
            ("statistics", "check"),
            ("stateful_followup", "followup"),
        ):
            with subtests.test(family=family, field=field):
                record = make_task(family, 1, 0, 183731, "test", 8)
                if field == "followup":
                    record["followup"]["check"] = (
                        "raise AssertionError('invalid followup')"
                    )
                else:
                    record[field] = "raise AssertionError('invalid answer')"
                with pytest.raises(AssertionError):
                    supervision(record, include_inspection=True, capture_rows=False)

    def test_templates_and_workspaces_are_disjoint_and_final_count_is_fixed(self):
        from model.train.src.generation import transfer
        from model.train.src.generation.transfer import prepare, supervision

        captured = {split: [] for split in ("train", "dev", "test")}

        def capture_rows(record, *args, **kwargs):
            rows = supervision(record, *args, **kwargs)
            captured[record["id"].split("-", 1)[0]].append(len(rows))
            return rows

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "corpus"
            with (
                patch(
                    "model.train.src.generation.transfer.supervision",
                    side_effect=capture_rows,
                ),
                redirect_stdout(io.StringIO()) as stdout,
            ):
                prepare(output)
            assert all(captured["train"])
            assert all(captured["dev"])
            assert captured["test"] == [0] * 120
            manifest = json.loads((output / "manifest.json").read_text())
            assert (output / "generator-frozen.py").read_bytes() == Path(transfer.__file__).read_bytes()
            assert all(path.stat().st_mode & 0o777 == 0o444 for path in output.iterdir())
            assert (output / "manifest.json").read_text() == json.dumps(manifest, indent=2) + "\n"
            assert stdout.getvalue() == json.dumps(manifest, indent=2) + "\n"
            assert manifest["splits"]["train"]["tasks"] == 144
            assert manifest["splits"]["dev"]["tasks"] == 60
            assert manifest["splits"]["test"]["tasks"] == 120
            assert manifest["splits"]["test"]["contrast_pairs"] == 60
            sets = [
                set(manifest["splits"][split]["template_ids"])
                for split in ("train", "dev", "test")
            ]
            assert [len(group) for group in sets] == [72, 24, 24]
            assert not (sets[0] & sets[1])
            assert not (sets[0] & sets[2])
            assert not (sets[1] & sets[2])
            assert not manifest["old_test_reused"]

    def test_snapshot_failure_does_not_publish_a_manifest(self, tmp_path):
        from model.train.src.generation.transfer import prepare

        output = tmp_path / "corpus"
        with (
            patch.object(Path, "write_bytes", side_effect=OSError("snapshot failed")),
            pytest.raises(OSError, match="snapshot failed"),
        ):
            prepare(output)
        assert not (output / "manifest.json").exists()
