"""Verify resume provenance and persistence without loading model weights."""

import json
import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest
import torch

from model.benchmark.src.corpus import sha256
from model.train.src.checkpoints import load_resume, save_checkpoint, sync_directory


class Artifact:
    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)


class TestCheckpoint:
    def test_resume_requires_checkpoint_local_metadata_even_when_parent_state_exists(
        self,
    ):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            (root / "train.jsonl").write_text('{"id":"first"}\n')
            state = {
                "train_sha256": sha256(root / "train.jsonl"),
                "accumulation": 4,
                "serialization": "raw",
                "updates": 99,
            }
            (root / "training.json").write_text(json.dumps(state))
            checkpoint = root / "checkpoint-4"
            checkpoint.mkdir()
            with pytest.raises(FileNotFoundError) as raised:
                load_resume(checkpoint, root, 4, "raw")
            assert raised.value.filename == str(checkpoint / "training.json")
            state["updates"] = 1
            (checkpoint / "training.json").write_text(json.dumps(state))
            assert load_resume(checkpoint, root, 4, "raw") == state

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
                with pytest.raises(OSError, match="disk failure"):
                    save_checkpoint(
                        Artifact(),
                        Artifact(),
                        optimizer,
                        {"updates": 2},
                        output,
                        "checkpoint-8",
                    )
            assert not (output / "checkpoint-8").exists()
            assert (output / "training.json").read_text() == previous
            assert sorted((p.name for p in output.iterdir())) == ["training.json"]

    def test_final_save_flushes_files_and_publishes_state_last(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            previous = '{"updates": 0}\n'
            (output / "training.json").write_text(previous)
            optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(2.0))])
            state = {"updates": 1}
            operations = Mock()

            def publish(source, destination):
                if destination == output / "training.json":
                    assert source.name == "training.json"
                    assert json.loads(source.read_text()) == state
                    assert (output / "training.json").read_text() == previous
                    assert (output / "optimizer.pt").is_file()
                os_replace(source, destination)

            os_replace = os.replace
            with (
                patch("model.train.src.checkpoints.os.fsync", wraps=os.fsync) as flush,
                patch(
                    "model.train.src.checkpoints.os.replace", side_effect=publish
                ) as replace,
                patch(
                    "model.train.src.checkpoints.sync_directory", wraps=sync_directory
                ) as sync,
                patch(
                    "model.train.src.checkpoints.tempfile.NamedTemporaryFile",
                    side_effect=AssertionError("Final metadata is already staged"),
                ),
            ):
                operations.attach_mock(replace, "replace")
                operations.attach_mock(sync, "sync")
                save_checkpoint(
                    Artifact(), Artifact(), optimizer, state, output, "final"
                )
            assert flush.call_count >= 5
            staged = replace.call_args_list[-1].args[0].parent
            assert operations.mock_calls == [
                call.sync(staged),
                call.replace(staged / "optimizer.pt", output / "optimizer.pt"),
                call.sync(output),
                call.replace(staged / "training.json", output / "training.json"),
                call.sync(output),
            ]
            assert (output / "optimizer.pt").is_file()
            assert json.loads((output / "training.json").read_text()) == state
            assert stat.S_IMODE((output / "training.json").stat().st_mode) == 384
            assert sorted((p.name for p in output.iterdir())) == [
                "optimizer.pt",
                "training.json",
            ]

    def test_failed_final_metadata_publication_preserves_state_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            previous = '{"updates": 1}\n'
            (output / "training.json").write_text(previous)
            optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(2.0))])
            os_replace = os.replace

            def fail_publication(source, destination):
                if destination == output / "training.json":
                    assert (output / "optimizer.pt").is_file()
                    assert json.loads(source.read_text()) == {"updates": 2}
                    raise OSError("metadata publication failed")
                os_replace(source, destination)

            with patch(
                "model.train.src.checkpoints.os.replace", side_effect=fail_publication
            ):
                with pytest.raises(OSError, match="metadata publication failed"):
                    save_checkpoint(
                        Artifact(),
                        Artifact(),
                        optimizer,
                        {"updates": 2},
                        output,
                        "final",
                    )
            assert (output / "training.json").read_text() == previous
            assert sorted((p.name for p in output.iterdir())) == [
                "optimizer.pt",
                "training.json",
            ]

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
            assert load_resume(checkpoint, root, 4, "raw") == state
            assert json.loads((output / "training.json").read_text()) == state
            restored = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(0.0))])
            restored.load_state_dict(
                torch.load(checkpoint / "optimizer.pt", weights_only=True)
            )
            assert restored.state_dict()["state"][0]["step"].item() == 1
            for accumulation, serialization in ((8, "raw"), (4, "chat")):
                with pytest.raises(
                    ValueError, match="gradient accumulation|different serialization"
                ):
                    load_resume(checkpoint, root, accumulation, serialization)
            (root / "train.jsonl").write_text('{"id":"changed"}\n')
            with pytest.raises(ValueError, match="dataset"):
                load_resume(checkpoint, root, 4, "raw")
