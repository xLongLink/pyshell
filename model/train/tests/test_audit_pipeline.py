"""Check shared loss encodings and independent checkpoint annotations."""

import json
from contextlib import ExitStack
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call, patch

import pytest


class Inputs(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = self["input_ids"]


@pytest.mark.skipif(
    not (
        all(find_spec(name) is not None for name in ("torch", "peft", "transformers"))
    ),
    reason="requires local ML extras: torch, peft, transformers",
)
class TestAuditPipeline:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def setup_class_data(cls):
        import torch

        from model.train.src import audit_pipeline

        cls.torch = torch
        cls.audit_pipeline = audit_pipeline

    @pytest.fixture(autouse=True)
    def setup(self):
        with ExitStack() as stack:
            torch = self.torch
            audit_pipeline = self.audit_pipeline
            temporary = stack.enter_context(TemporaryDirectory(dir="/tmp/opencode"))
            root = Path(temporary)
            self.output = root / "output"
            benchmark = root / "benchmark"
            data = root / "train/data/raw"
            data.mkdir(parents=True)
            self.known = [self.record(f"known-{index}") for index in range(2)]
            self.heldout = [self.record(f"heldout-{index}") for index in range(3)]
            for name, records in (("train", self.known), ("validation", self.heldout)):
                (data / f"{name}.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records)
                )
            stack.enter_context(patch.object(audit_pipeline, "ROOT", benchmark))
            stack.enter_context(
                patch.object(audit_pipeline, "TRAIN_ROOT", root / "train")
            )
            stack.enter_context(patch.object(audit_pipeline.torch, "set_num_threads"))
            stack.enter_context(
                patch.object(audit_pipeline.torch, "set_num_interop_threads")
            )
            stack.enter_context(patch("builtins.print"))
            self.sample = stack.enter_context(
                patch.object(
                    audit_pipeline,
                    "balanced_sample",
                    side_effect=[self.known, self.heldout],
                )
            )
            self.tokenizer = MagicMock()
            self.tokenizer.encode.return_value = [11, 12]
            self.tokenizer.return_value = Inputs(torch.tensor([[1, 2]]))
            self.tokenizer.decode.side_effect = lambda ids, **kwargs: (
                f"value = {ids[0].item()}\n{audit_pipeline.END_CODE}"
            )
            stack.enter_context(
                patch.object(
                    audit_pipeline.AutoTokenizer,
                    "from_pretrained",
                    return_value=self.tokenizer,
                )
            )
            self.base_load = stack.enter_context(
                patch.object(
                    audit_pipeline.AutoModelForImageTextToText, "from_pretrained"
                )
            )
            self.models = {}
            for label, value in (("best600", 100), ("stopped", 200)):
                model = MagicMock()
                model.eval.return_value = model
                model.generate.return_value = torch.tensor([[1, 2, value]])
                self.models[label] = model
            self.adapter_load = stack.enter_context(
                patch.object(
                    audit_pipeline.PeftModel,
                    "from_pretrained",
                    side_effect=self.load_adapter,
                )
            )
            self.encoded = []
            self.encode = stack.enter_context(
                patch.object(audit_pipeline, "encode", side_effect=self.encode_record)
            )
            self.loss = stack.enter_context(
                patch.object(
                    audit_pipeline, "loss_for", side_effect=self.checkpoint_loss
                )
            )
            self.worker = stack.enter_context(
                patch.object(
                    audit_pipeline.subprocess, "run", side_effect=self.run_worker
                )
            )
            yield

    @staticmethod
    def record(identifier):
        return {
            "id": identifier,
            "prompt": f"prompt {identifier}",
            "completion": "value = 1",
        }

    def load_adapter(self, base, checkpoint):
        return self.models[
            "best600" if checkpoint.name == "checkpoint-600" else "stopped"
        ]

    def encode_record(self, record, tokenizer, max_length, serialization):
        self.base_load.assert_not_called()
        assert tokenizer is self.tokenizer
        assert (max_length, serialization) == (2048, "raw")
        assert "hf_target_loss" not in record
        item = {"id": record["id"], "position": len(self.encoded)}
        self.encoded.append(item)
        return item

    def checkpoint_loss(self, model, item):
        assert item is self.encoded[item["position"]]
        return (100 if model is self.models["best600"] else 200) + item["position"]

    def run_worker(self, command, **kwargs):
        if "--audit-output" in command:
            output = Path(command[command.index("--audit-output") + 1])
            output.mkdir()
            (output / "summary.json").write_text(json.dumps({"label": output.name}))
        return MagicMock(returncode=0)

    def run_audit(self, hf_samples=1, only=None):
        argv = [
            "--output",
            str(self.output),
            "--known-samples",
            "2",
            "--heldout-samples",
            "3",
            "--hf-samples",
            str(hf_samples),
        ]
        if only:
            argv.extend(["--only", only])
        self.audit_pipeline.main(argv)

    def examples(self, label):
        return [
            json.loads(line)
            for line in (self.output / f"{label}-examples.jsonl")
            .read_text()
            .splitlines()
        ]

    def test_encodes_each_sample_position_once_for_two_checkpoints(self):
        self.run_audit()
        assert [item["id"] for item in self.encoded] == ["known-0", "heldout-0"]
        assert self.encode.call_count == 2
        assert self.loss.call_args_list == [
            call(model, item) for model in self.models.values() for item in self.encoded
        ]
        assert self.base_load.call_count == 2
        for label, value in (("best600", 100), ("stopped", 200)):
            examples = self.examples(label)
            assert [record["id"] for record in examples] == [
                record["id"] for record in self.known + self.heldout
            ]
            assert [record["audit_split"] for record in examples] == ["known"] * 2 + [
                "heldout"
            ] * 3
            assert [record.get("hf_target_loss") for record in examples] == [
                value,
                None,
                value + 1,
                None,
                None,
            ]
            assert examples[0]["hf_prediction"]["source"] == f"value = {value}\n"
            assert examples[0]["hf_prediction"]["end_marker_seen"]
            assert all("hf_prediction" not in record for record in examples[1:])
            assert all(record["token_ids"] == [11, 12] for record in examples)
            assert self.models[label].generate.call_args.kwargs == {
                "input_ids": self.tokenizer.return_value.input_ids,
                "do_sample": False,
                "max_new_tokens": 192,
                "stop_strings": [self.audit_pipeline.END_CODE],
                "tokenizer": self.tokenizer,
            }
        assert json.loads((self.output / "summary.json").read_text()) == {
            "best600": {"label": "best600"},
            "stopped": {"label": "stopped"},
        }
        assert all(
            "hf_target_loss" not in record for record in self.known + self.heldout
        )

    def test_overlapping_slices_keep_duplicate_positions_and_last_annotations(self):
        self.run_audit(hf_samples=3)
        assert [item["id"] for item in self.encoded] == [
            "known-0",
            "known-1",
            "heldout-0",
            "heldout-0",
            "heldout-1",
            "heldout-2",
        ]
        assert self.encode.call_count == 6
        assert self.loss.call_args_list == [
            call(model, item) for model in self.models.values() for item in self.encoded
        ]
        for label, value in (("best600", 100), ("stopped", 200)):
            examples = self.examples(label)
            assert [record["hf_target_loss"] for record in examples] == [
                value,
                value + 1,
                value + 3,
                value + 4,
                value + 5,
            ]
            assert self.models[label].generate.call_count == 3
            assert all("hf_prediction" in record for record in examples[:3])
            assert all("hf_prediction" not in record for record in examples[3:])

    def test_oversize_error_precedes_model_loading(self):
        self.encode.side_effect = ValueError(
            "known-0 needs 2049 tokens, above 2048; refusing to truncate supervision"
        )
        with pytest.raises(ValueError, match="refusing to truncate supervision"):
            self.run_audit()
        self.base_load.assert_not_called()
        self.adapter_load.assert_not_called()
        self.loss.assert_not_called()
        self.worker.assert_not_called()
        assert list(self.output.iterdir()) == []

    def test_only_selected_checkpoint_is_loaded_and_annotated(self):
        self.run_audit(only="stopped")
        assert self.encode.call_count == 2
        assert self.loss.call_args_list == [
            call(self.models["stopped"], item) for item in self.encoded
        ]
        self.base_load.assert_called_once()
        self.adapter_load.assert_called_once()
        assert self.adapter_load.call_args.args[1].name == "interrupted"
        self.models["best600"].generate.assert_not_called()
        assert not (self.output / "best600-examples.jsonl").exists()
        assert [
            record.get("hf_target_loss") for record in self.examples("stopped")
        ] == [200, None, 201, None, None]
        assert json.loads((self.output / "summary.json").read_text()) == {
            "stopped": {"label": "stopped"}
        }
