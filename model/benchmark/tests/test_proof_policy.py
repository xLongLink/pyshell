from model.benchmark.src.proof_policy import final_passed, development_ready, acceptance_metadata


class TestProofPolicy:
    def test_acceptance_metadata_preserves_order_and_returns_a_fresh_dict(self):
        metadata = acceptance_metadata()
        assert list(metadata.items()) == [
            ("test_tasks", 120),
            ("required_passes", 97),
            ("strictly_greater_than", 0.8),
            ("max_calls_per_stage", 4),
        ]
        metadata["required_passes"] = 0
        assert acceptance_metadata()["required_passes"] == 97

    def test_development_requires_complete_overall_and_family_minimums(self, subtests):
        summary = {"complete": True, "tasks": 60, "success_rate": 0.9, "by_family": {"recovery": {"tasks": 5, "passed": 3}}}
        assert development_ready(summary)
        for override in (
            {"complete": False},
            {"tasks": 59},
            {"success_rate": 0.89},
            {"by_family": {"recovery": {"tasks": 5, "passed": 2}}},
        ):
            with subtests.test(override=override):
                assert not development_ready({**summary, **override})

    def test_final_requires_the_full_set_and_strictly_more_than_eighty_percent(self):
        assert final_passed(120, 97)
        assert not final_passed(120, 96)
        assert not final_passed(100, 97)
