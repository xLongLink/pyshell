"""Check explicit device selection and bounded resume budgets."""

import io
import unittest
from contextlib import redirect_stderr

from model.train.src.training import parse_args


class TrainingArgumentsTests(unittest.TestCase):
    def test_cpu_remains_default(self):
        self.assertEqual(parse_args([]).device, "cpu")

    def test_smoke_stop_preserves_original_schedule(self):
        args = parse_args(
            ["--device", "cuda", "--steps", "1200", "--stop-after", "804"]
        )
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.steps, 1200)
        self.assertEqual(args.stop_after, 804)

    def test_stop_requires_an_optimizer_boundary_within_the_budget(self):
        for stop in ("0", "803", "1204"):
            with (
                self.subTest(stop=stop),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(["--steps", "1200", "--stop-after", stop])
