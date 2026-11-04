"""Check balanced inspection and recovery supervision without a model."""

import unittest
from importlib.util import find_spec


@unittest.skipUnless(find_spec("pandas"), "Install the benchmark data extra")
class ExecutedCurriculumTests(unittest.TestCase):
    def test_every_family_and_variant_has_supervised_inspection_and_a_verified_answer(self):
        from dataset.benchmark.src.reference import ReferenceWorkspace
        from dataset.generation.src.executed import trajectory, training_tasks
        from dataset.benchmark.src.task_types import FAMILIES

        records = list(training_tasks(2, 917341))
        self.assertEqual(len(records), len(FAMILIES) * 4)
        for record in records:
            with self.subTest(task=record["id"]):
                rows = trajectory(record, ReferenceWorkspace())
                self.assertEqual(rows[0]["completion"], record["inspect"])
                self.assertIn(record["code"], [row["completion"] for row in rows])
                self.assertIn(record["template_id"].rsplit("t", 1)[1], {"0", "1"})
                for row in rows:
                    self.assertNotIn(record["check"], row["prompt"])

    def test_sum_and_mean_both_receive_direct_and_error_recovery_trajectories(self):
        from dataset.benchmark.src.reference import ReferenceWorkspace
        from dataset.generation.src.executed import trajectory, training_tasks

        records = [record for record in training_tasks(1, 917341) if record["family"] == "recover_execution"]
        self.assertEqual({record["variant"] for record in records}, {0, 1})
        for record in records:
            direct = trajectory(record, ReferenceWorkspace())
            retry = trajectory(record, ReferenceWorkspace(), retry=True)
            self.assertEqual(len(direct), 2)
            self.assertEqual(len(retry), 3)
            self.assertIn("RuntimeError", retry[-1]["prompt"])
            self.assertIn(".recover()", direct[-1]["completion"])
            self.assertIn(".recover()", retry[-1]["completion"])
