"""Keep boundary matching resident without changing generation or parsing."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from dataset.benchmark.src.serialization import CODE, CONSOLE, END_CODE, TASK
from model.train.src.inference import HFCompletion


class Inputs(dict):
    def __init__(self):
        super().__init__(input_ids=torch.tensor([[1, 2]]))
        self.input_ids = self["input_ids"]

    def to(self, device):
        return self


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MagicMock()
        self.tokenizer.return_value = Inputs()
        self.model = MagicMock(device=torch.device("cpu"))
        self.model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model

    def completion(self):
        with (
            patch("model.train.src.inference.torch.set_num_interop_threads"),
            patch("model.train.src.inference.torch.set_num_threads"),
            patch(
                "model.train.src.inference.AutoTokenizer.from_pretrained",
                return_value=self.tokenizer,
            ),
            patch(
                "model.train.src.inference.AutoModelForImageTextToText.from_pretrained",
                return_value=self.model,
            ),
            patch(
                "model.train.src.inference.StopStringCriteria",
                return_value=SimpleNamespace(),
            ) as build,
        ):
            completion = HFCompletion()
            build.assert_called_once_with(
                self.tokenizer, [END_CODE, TASK, CONSOLE, CODE]
            )
        return completion

    def test_boundary_matcher_is_built_once_and_reused_across_requests(self):
        completion = self.completion()
        self.tokenizer.decode.return_value = "result = 1\n" + END_CODE
        streamer = object()
        for _ in range(2):
            self.assertEqual(
                completion("prompt", streamer=streamer), ("result = 1\n", True)
            )
            options = self.model.generate.call_args.kwargs
            self.assertIs(options["stopping_criteria"], completion.stopping_criteria)
            self.assertIsNone(options["stop_strings"])
            self.assertIs(options["streamer"], streamer)
            self.assertEqual(options["max_new_tokens"], 384)
            self.assertFalse(options["do_sample"])

    def test_earliest_boundary_and_missing_marker_keep_existing_semantics(self):
        completion = self.completion()
        for text, expected in (
            ("code" + TASK + END_CODE, ("code", False)),
            ("code" + END_CODE + CONSOLE, ("code", True)),
            ("code", ("code", False)),
        ):
            with self.subTest(text=text):
                self.tokenizer.decode.return_value = text
                self.assertEqual(completion("prompt"), expected)
