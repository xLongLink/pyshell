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
    @pytest.mark.parametrize("name", ["checkpoint-4", "final"])
    def test_save_requires_existing_output_directory(self, tmp_path, name):
        output = tmp_path / "missing" / "adapter"
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(2.0))])
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=4)
        with pytest.raises(FileNotFoundError):
            save_checkpoint(
                Artifact(),
                Artifact(),
                optimizer,
                {"updates": 1},
                output,
                name,
                scheduler=scheduler,
            )
        assert not output.parent.exists()
        assert list(tmp_path.iterdir()) == []

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
                load_resume(checkpoint, root, 4)
            assert raised.value.filename == str(checkpoint / "training.json")
            state["updates"] = 1
            (checkpoint / "training.json").write_text(json.dumps(state))
            assert load_resume(checkpoint, root, 4) == state

    @pytest.mark.parametrize("failed_file", ["optimizer.pt", "scheduler.pt"])
    def test_failed_save_never_publishes_a_checkpoint_or_advances_state(
        self, failed_file
    ):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            previous = '{"updates": 1}\n'
            (output / "training.json").write_text(previous)
            optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(2.0))])
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=4)
            torch_save = torch.save

            def fail_save(state, path):
                if path.name == failed_file:
                    raise OSError("disk failure")
                torch_save(state, path)

            with patch(
                "model.train.src.checkpoints.torch.save",
                side_effect=fail_save,
            ):
                with pytest.raises(OSError, match="disk failure"):
                    save_checkpoint(
                        Artifact(),
                        Artifact(),
                        optimizer,
                        {"updates": 2},
                        output,
                        "checkpoint-8",
                        scheduler=scheduler,
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
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=4)
            state = {"updates": 1}
            operations = Mock()
            flushed_inodes = set()
            os_fsync = os.fsync

            def flush_file(descriptor):
                os_fsync(descriptor)
                flushed_inodes.add(os.fstat(descriptor).st_ino)

            def publish(source, destination):
                if destination == output / "training.json":
                    assert source.name == "training.json"
                    assert json.loads(source.read_text()) == state
                    assert (output / "training.json").read_text() == previous
                    assert (output / "optimizer.pt").is_file()
                    assert (
                        torch.load(output / "scheduler.pt", weights_only=True)
                        == scheduler.state_dict()
                    )
                    assert (output / "scheduler.pt").stat().st_ino in flushed_inodes
                os_replace(source, destination)

            os_replace = os.replace
            with (
                patch(
                    "model.train.src.checkpoints.os.fsync", side_effect=flush_file
                ) as flush,
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
                    Artifact(),
                    Artifact(),
                    optimizer,
                    state,
                    output,
                    "final",
                    scheduler=scheduler,
                )
            assert flush.call_count >= 6
            staged = replace.call_args_list[-1].args[0].parent
            assert operations.mock_calls[0] == call.sync(staged)
            assert sorted(
                operations.mock_calls[1:3], key=lambda operation: operation.args[0].name
            ) == [
                call.replace(staged / "optimizer.pt", output / "optimizer.pt"),
                call.replace(staged / "scheduler.pt", output / "scheduler.pt"),
            ]
            assert operations.mock_calls[3:] == [
                call.sync(output),
                call.replace(staged / "training.json", output / "training.json"),
                call.sync(output),
            ]
            assert (output / "optimizer.pt").is_file()
            assert json.loads((output / "training.json").read_text()) == state
            assert stat.S_IMODE((output / "training.json").stat().st_mode) == 384
            assert sorted((p.name for p in output.iterdir())) == [
                "optimizer.pt",
                "scheduler.pt",
                "training.json",
            ]

    def test_failed_final_metadata_publication_preserves_state_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            previous = '{"updates": 1}\n'
            (output / "training.json").write_text(previous)
            optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(2.0))])
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=4)
            os_replace = os.replace

            def fail_publication(source, destination):
                if destination == output / "training.json":
                    assert (output / "optimizer.pt").is_file()
                    assert (
                        torch.load(output / "scheduler.pt", weights_only=True)
                        == scheduler.state_dict()
                    )
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
                        scheduler=scheduler,
                    )
            assert (output / "training.json").read_text() == previous
            assert sorted((p.name for p in output.iterdir())) == [
                "optimizer.pt",
                "scheduler.pt",
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
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=4)
            parameter.square().backward()
            optimizer.step()
            scheduler.step()
            output = root / "adapter"
            output.mkdir()
            save_checkpoint(
                Artifact(),
                Artifact(),
                optimizer,
                state,
                output,
                "checkpoint-4",
                scheduler=scheduler,
            )
            checkpoint = output / "checkpoint-4"
            assert load_resume(checkpoint, root, 4) == state
            assert json.loads((output / "training.json").read_text()) == state
            restored = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(0.0))])
            restored.load_state_dict(
                torch.load(checkpoint / "optimizer.pt", weights_only=True)
            )
            assert restored.state_dict()["state"][0]["step"].item() == 1
            with pytest.raises(ValueError, match="gradient accumulation"):
                load_resume(checkpoint, root, 8)
            (root / "train.jsonl").write_text('{"id":"changed"}\n')
            with pytest.raises(ValueError, match="dataset"):
                load_resume(checkpoint, root, 4)

    @pytest.mark.parametrize("name", ["checkpoint-4", "final"])
    def test_checkpoint_preserves_scheduler_position_and_next_learning_rate(
        self, tmp_path, name
    ):
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=4)
        parameter.square().backward()
        optimizer.step()
        scheduler.step()
        save_checkpoint(
            Artifact(),
            Artifact(),
            optimizer,
            {"updates": 1},
            tmp_path,
            name,
            scheduler=scheduler,
        )
        checkpoint = tmp_path if name == "final" else tmp_path / name
        restored_optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.tensor(0.0))], lr=0.01
        )
        restored_scheduler = torch.optim.lr_scheduler.LinearLR(
            restored_optimizer, total_iters=4
        )
        restored_optimizer.load_state_dict(
            torch.load(checkpoint / "optimizer.pt", weights_only=True)
        )
        restored_scheduler.load_state_dict(
            torch.load(checkpoint / "scheduler.pt", weights_only=True)
        )
        assert restored_scheduler.state_dict() == scheduler.state_dict()
        optimizer.step()
        scheduler.step()
        restored_optimizer.step()
        restored_scheduler.step()
        assert (
            restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
        )

    @pytest.mark.parametrize("serialization", [None, "chat", "unknown"])
    def test_resume_rejects_checkpoints_without_raw_protocol_metadata(
        self, tmp_path, serialization
    ):
        (tmp_path / "train.jsonl").write_text('{"id":"first"}\n')
        state = {"train_sha256": sha256(tmp_path / "train.jsonl"), "accumulation": 4}
        if serialization is not None:
            state["serialization"] = serialization
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "training.json").write_text(json.dumps(state))
        with pytest.raises(ValueError, match="raw protocol serialization"):
            load_resume(checkpoint, tmp_path, 4)
