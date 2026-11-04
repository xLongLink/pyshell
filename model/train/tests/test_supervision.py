"""Verify causal target alignment and loss masking without loading model weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from dataset.benchmark.src.serialization import END_CODE
from model.train.src.supervision import encode, evaluate_loss, loss_for


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(char) for char in text]


class Model(torch.nn.Module):
    def __init__(self, length):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(1, length, 128))

    def forward(self, *, input_ids, use_cache, logits_to_keep):
        assert not use_cache
        assert input_ids.device == self.logits.device
        assert logits_to_keep.device == self.logits.device
        return SimpleNamespace(logits=self.logits[:, logits_to_keep])


class SupervisionTests(unittest.TestCase):
    def test_raw_training_has_no_chat_template_and_teaches_end_marker(self):
        record = {
            "id": "example",
            "prompt": "<|task|>\nSum values.\n<|code|>\n",
            "completion": "result = sum(values)",
        }
        item = encode(record, Tokenizer(), 1024, "raw")
        self.assertEqual(
            item["ids"][: item["prefix_length"]],
            [ord(char) for char in record["prompt"]],
        )
        self.assertEqual(
            item["ids"][item["prefix_length"] :],
            [ord(char) for char in record["completion"] + "\n" + END_CODE],
        )

    def test_only_causally_aligned_target_logits_are_supervised(self):
        item = {"ids": [5, 8, 13, 21, 34, 55], "prefix_length": 3}
        model = Model(len(item["ids"]))
        loss = loss_for(model, item)
        expected = F.cross_entropy(
            model.logits[:, 2:5].reshape(-1, 128), torch.tensor([21, 34, 55])
        )
        self.assertTrue(torch.equal(loss, expected))
        loss.backward()
        self.assertEqual(torch.count_nonzero(model.logits.grad[:, :2]).item(), 0)
        self.assertEqual(torch.count_nonzero(model.logits.grad[:, 5:]).item(), 0)
        self.assertGreater(torch.count_nonzero(model.logits.grad[:, 2:5]).item(), 0)

    def test_overlength_examples_are_rejected_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            encode(
                {"id": "too-long", "prompt": "abc", "completion": "def"},
                Tokenizer(),
                3,
                "raw",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_loss_and_backward_keep_tensors_on_model_device(self):
        model = Model(6).to("cuda")
        loss = loss_for(model, {"ids": [5, 8, 13, 21, 34, 55], "prefix_length": 3})
        self.assertEqual(loss.device.type, "cuda")
        loss.backward()
        self.assertTrue(torch.isfinite(model.logits.grad).all())

    def test_validation_loss_is_weighted_by_target_length(self):
        items = [
            {"ids": [1, 2, 3], "prefix_length": 2},
            {"ids": [1, 2, 3, 4], "prefix_length": 1},
        ]
        model = Model(4)
        with patch(
            "model.train.src.supervision.loss_for",
            side_effect=[torch.tensor(2.0), torch.tensor(6.0)],
        ):
            self.assertEqual(evaluate_loss(model, items), 5.0)
        self.assertTrue(model.training)
