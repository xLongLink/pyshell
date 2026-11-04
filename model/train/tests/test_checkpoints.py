"""Verify resume provenance and persistence without loading model weights."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from dataset.benchmark.src.corpus import sha256
from model.train.src.checkpoints import load_resume, save_checkpoint


class Artifact:
    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)


class CheckpointTests(unittest.TestCase):
    def test_failed_save_never_publishes_a_checkpoint_or_advances_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            previous = '{"updates": 1}\n'
            (output / "training.json").write_text(previous)
            optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(2.0))])
            with patch(
                "model.train.src.checkpoints.torch.save",
                side_effect=OSError("disk failure"),
            ):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    save_checkpoint(
                        Artifact(),
                        Artifact(),
                        optimizer,
                        {"updates": 2},
                        output,
                        "checkpoint-8",
                    )
            self.assertFalse((output / "checkpoint-8").exists())
            self.assertEqual((output / "training.json").read_text(), previous)
            self.assertEqual(
                sorted(p.name for p in output.iterdir()), ["training.json"]
            )

    def test_final_save_flushes_files_and_publishes_state_last(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(2.0))])
            state = {"updates": 1}
            with (
                patch("model.train.src.checkpoints.os.fsync", wraps=os.fsync) as flush,
                patch(
                    "model.train.src.checkpoints.os.replace", wraps=os.replace
                ) as replace,
            ):
                save_checkpoint(
                    Artifact(), Artifact(), optimizer, state, output, "final"
                )
            self.assertGreaterEqual(flush.call_count, 5)
            self.assertEqual(
                replace.call_args_list[-1].args[1], output / "training.json"
            )
            self.assertTrue((output / "optimizer.pt").is_file())
            self.assertEqual(json.loads((output / "training.json").read_text()), state)

    def test_saved_checkpoint_restores_provenance_and_optimizer_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.jsonl").write_text('{"id":"first"}\n')
            state = {
                "train_sha256": sha256(root / "train.jsonl"),
                "accumulation": 4,
                "serialization": "raw",
                "updates": 1,
            }
            parameter = torch.nn.Parameter(torch.tensor(2.0))
            optimizer = torch.optim.AdamW([parameter])
            parameter.square().backward()
            optimizer.step()
            output = root / "adapter"
            save_checkpoint(
                Artifact(), Artifact(), optimizer, state, output, "checkpoint-4"
            )
            checkpoint = output / "checkpoint-4"
            self.assertEqual(load_resume(checkpoint, root, 4, "raw"), state)
            self.assertEqual(json.loads((output / "training.json").read_text()), state)
            restored = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(0.0))])
            restored.load_state_dict(
                torch.load(checkpoint / "optimizer.pt", weights_only=True)
            )
            self.assertEqual(restored.state_dict()["state"][0]["step"].item(), 1)
            for accumulation, serialization in ((8, "raw"), (4, "chat")):
                with self.assertRaises(ValueError):
                    load_resume(checkpoint, root, accumulation, serialization)
            (root / "train.jsonl").write_text('{"id":"changed"}\n')
            with self.assertRaisesRegex(ValueError, "dataset"):
                load_resume(checkpoint, root, 4, "raw")
