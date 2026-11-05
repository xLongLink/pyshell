"""Check binding fixtures and negative controls independently of model output."""

from collections import defaultdict

import pytest

from model.benchmark.src.reference import ReferenceWorkspace
from model.benchmark.src.task_types import FAMILIES
from model.train.src.generation.binding import make_task, verify_negative_control
from model.train.src.generation.binding_curriculum import binding_record, trajectories
from model.train.src.generation.binding_evaluation import evaluation_task
from model.train.src.generation.binding_language import (
    diversify_binding,
    diversify_support,
)
from model.train.src.generation.join_curriculum import join_record
from model.train.src.generation.transfer import make_contrast_pair, supervision


class TestBinding:
    @pytest.mark.parametrize("error_count", [0, 1, 2])
    def test_trajectory_ids_and_targets_preserve_direct_and_recovery_order(
        self, error_count
    ):
        record, errors = binding_record("joined_totals", 0, 0, 719)
        workspace = ReferenceWorkspace()
        rows = trajectories(record, errors[:error_count], workspace)
        assert [row["id"] for row in rows] == [
            record["id"] + "-inspect",
            record["id"] + "-solve",
            *[record["id"] + f"-recover-{index}" for index in range(error_count)],
        ]
        assert [row["completion"] for row in rows] == [
            record["inspect"],
            *[record["code"] for _ in range(error_count + 1)],
        ]

    def test_negative_control_verifier_executes_in_a_fresh_workspace(self):
        record = {
            "id": "negative-control",
            "setup": "value = 7",
            "negative_control": "result = 0",
            "check": "assert result == value",
        }
        workspace = ReferenceWorkspace()
        workspace.execute("value = 99", "reset")
        verify_negative_control(record, workspace)
        assert workspace.namespace["value"] == 7
        assert workspace.namespace["result"] == 0

    @pytest.mark.parametrize(
        "field,source",
        [
            ("setup", "raise ValueError('fixture failed')"),
            ("negative_control", "raise ValueError('control failed')"),
            ("negative_control", "result = value"),
            ("check", "pass"),
        ],
    )
    def test_negative_control_verifier_rejects_invalid_controls(self, field, source):
        record = {
            "id": "negative-control",
            "setup": "value = 7",
            "negative_control": "result = 0",
            "check": "assert result == value",
            field: source,
        }
        workspace = ReferenceWorkspace()
        with pytest.raises(AssertionError):
            verify_negative_control(record, workspace)

    def test_explicit_join_selection_preserves_contrast_and_order(self):
        workspace = ReferenceWorkspace()
        for pair in range(12):
            records = make_contrast_pair("left_join", pair, 1307, "train", pair % 6)
            for record in records:
                record = join_record(record)
                assert supervision(record, True, workspace=workspace)
            self.execute(workspace, records[0]["setup"], True)
            self.execute(workspace, records[0]["code"])
            assert workspace.execute(records[1]["check"])["exception"] is not None

    def test_language_styles_do_not_determine_api_order_or_schema_shape(self):
        orders, keywords, shapes = defaultdict(set), defaultdict(set), defaultdict(set)
        for pair in range(64):
            for family in ("inspect_callable", "joined_totals"):
                record, _ = binding_record(family, pair, 0, 20261117)
                record = diversify_binding(record, pair)
                axes = record["binding_axes"]
                if family == "inspect_callable":
                    orders[record["template_id"]].add(axes["order"])
                    keywords[tuple(sorted(axes["keywords"]))].add(axes["order"])
                else:
                    shapes[record["template_id"]].add(
                        (axes["nested_rows"], axes["nested_prices"])
                    )
        assert len(orders) == len(keywords) == len(shapes) == 6
        assert all(len(values) == 2 for values in orders.values())
        assert all(len(values) == 2 for values in keywords.values())
        assert all(len(values) == 4 for values in shapes.values())

    def test_language_variants_preserve_binding_results(self):
        workspace = ReferenceWorkspace()
        for family in ("inspect_callable", "joined_totals"):
            for pair in range(12):
                record, errors = binding_record(family, pair, pair % 2, 1307)
                record = diversify_binding(record, pair)
                assert len(trajectories(record, errors, workspace)) == 4

    def test_support_paraphrases_preserve_join_and_followup_contracts(self):
        workspace = ReferenceWorkspace()
        for family in ("left_join", "stateful_followup", "missing_input"):
            for pair in range(6):
                for record in make_contrast_pair(family, pair, 1307, "train", pair % 6):
                    record = diversify_support(record, pair)
                    assert supervision(record, True, workspace=workspace)

    def test_role_binding_direct_and_recovery_targets_execute(self):
        workspace = ReferenceWorkspace()
        for family in ("inspect_callable", "joined_totals"):
            for pair in range(8):
                record, errors = binding_record(family, pair, pair % 2, 719)
                rows = trajectories(record, errors, workspace)
                assert len(rows) == 4
                assert all(row["completion"] not in errors for row in rows)
                assert '"exception": "Traceback' in rows[-1]["prompt"]
                assert record["check"] not in rows[-1]["prompt"]

    def test_final_shape_schedule_covers_both_orders_and_all_schema_shapes(self):
        for seed in (1, 719, 20261107):
            orders = {
                make_task("inspect_callable", pair, 0, seed, "test")["binding_axes"][
                    "order"
                ]
                for pair in range(5)
            }
            shapes = {
                (
                    record["binding_axes"]["nested_rows"],
                    record["binding_axes"]["nested_prices"],
                )
                for pair in range(5)
                for record in [make_task("joined_totals", pair, 0, seed, "test")]
            }
            assert orders == {"add_then_multiply", "multiply_then_add"}
            assert shapes == {
                (False, False),
                (False, True),
                (True, False),
                (True, True),
            }

    def test_all_evaluation_family_references(self):
        workspace = ReferenceWorkspace()
        for split in ("dev", "test"):
            for family in FAMILIES:
                for variant in (0, 1):
                    record = evaluation_task(family, 1, variant, 17397, split)
                    self.execute(workspace, record["setup"], True)
                    self.execute(workspace, record["code"])
                    self.execute(workspace, record["check"])
                    if record["followup"]:
                        self.execute(workspace, "saved_result = result")
                        self.execute(workspace, record["followup"]["code"])
                        self.execute(workspace, record["followup"]["check"])

    def execute(self, workspace, source, reset=False):
        result = workspace.execute(source, "reset" if reset else "execute")
        assert result["exception"] is None, result["exception"]

    def test_reference_and_negative_controls_across_shapes(self):
        axes = set()
        workspace = ReferenceWorkspace()
        for family in ("inspect_callable", "joined_totals"):
            for pair in range(40):
                for variant in (0, 1):
                    record = make_task(family, pair, variant, 719, "train")
                    self.execute(workspace, record["setup"], True)
                    self.execute(workspace, record["inspect"])
                    self.execute(workspace, record["code"])
                    self.execute(workspace, record["check"])
                    self.execute(workspace, record["setup"], True)
                    self.execute(workspace, record["negative_control"])
                    assert workspace.execute(record["check"])["exception"] is not None
                    if family == "joined_totals":
                        axes.add(
                            (
                                record["binding_axes"]["nested_rows"],
                                record["binding_axes"]["nested_prices"],
                            )
                        )
        assert axes == {(False, False), (False, True), (True, False), (True, True)}

    def test_pairs_share_workspace_but_require_different_answers(self):
        workspace = ReferenceWorkspace()
        for family in ("inspect_callable", "joined_totals"):
            for pair in range(20):
                records = [
                    make_task(family, pair, variant, 971, "train") for variant in (0, 1)
                ]
                assert records[0]["setup"] == records[1]["setup"]
                results = []
                for record in records:
                    self.execute(workspace, record["setup"], True)
                    self.execute(workspace, record["code"])
                    results.append(workspace.namespace["result"])
                assert results[0] != results[1]
                self.execute(workspace, records[0]["setup"], True)
                self.execute(workspace, records[1]["code"])
                assert workspace.execute(records[0]["check"])["exception"] is not None

    def test_api_and_schema_axes_are_not_bound_to_wording_split(self):
        for family in ("inspect_callable", "joined_totals"):
            for pair in range(10):
                records = [
                    make_task(family, pair, 0, 179, split)
                    for split in ("train", "dev", "test")
                ]
                assert len({r["setup"] for r in records}) == 1
                assert len({r["template_id"] for r in records}) == 3
                assert len({r["instruction"] for r in records}) == 3

    def test_invalid_task_request_rejected(self):
        for family, variant, split in (
            ("unknown", 0, "train"),
            ("joined_totals", 2, "train"),
            ("inspect_callable", 0, "unknown"),
        ):
            with pytest.raises(ValueError, match="Unsupported binding task"):
                make_task(family, 0, variant, 1, split)
