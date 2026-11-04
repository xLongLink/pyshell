"""Check shared loss encodings and independent checkpoint annotations."""

import json
import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call, patch


class Inputs(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = self["input_ids"]


@unittest.skipUnless(
    all(find_spec(name) is not None for name in ("torch", "peft", "transformers")),
    "requires local ML extras: torch, peft, transformers",
)
class AuditPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from dataset.benchmark.src import audit_pipeline

        cls.torch = torch
        cls.audit_pipeline = audit_pipeline

    def setUp(self):
        torch = self.torch
        audit_pipeline = self.audit_pipeline
        temporary = self.enterContext(TemporaryDirectory(dir="/tmp/opencode"))
        root = Path(temporary)
        self.output = root / "output"
        benchmark = root / "dataset/benchmark"
        data = benchmark.parent / "data"
        data.mkdir(parents=True)
        self.known = [self.record(f"known-{index}") for index in range(2)]
        self.heldout = [self.record(f"heldout-{index}") for index in range(3)]
        for name, records in (("train", self.known), ("validation", self.heldout)):
            (data / f"{name}.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
        self.enterContext(patch.object(audit_pipeline, "ROOT", benchmark))
        self.enterContext(patch.object(audit_pipeline, "TRAIN_ROOT", root / "model/train"))
        self.enterContext(patch.object(audit_pipeline.torch, "set_num_threads"))
        self.enterContext(patch.object(audit_pipeline.torch, "set_num_interop_threads"))
        self.enterContext(patch("builtins.print"))
        self.sample = self.enterContext(patch.object(audit_pipeline, "balanced_sample", side_effect=[self.known, self.heldout]))
        self.tokenizer = MagicMock()
        self.tokenizer.encode.return_value = [11, 12]
        self.tokenizer.return_value = Inputs(torch.tensor([[1, 2]]))
        self.tokenizer.decode.side_effect = lambda ids, **kwargs: f"value = {ids[0].item()}\n{audit_pipeline.END_CODE}"
        self.enterContext(patch.object(audit_pipeline.AutoTokenizer, "from_pretrained", return_value=self.tokenizer))
        self.base_load = self.enterContext(patch.object(audit_pipeline.AutoModelForImageTextToText, "from_pretrained"))
        self.models = {}
        for label, value in (("best600", 100), ("stopped", 200)):
            model = MagicMock()
            model.eval.return_value = model
            model.generate.return_value = torch.tensor([[1, 2, value]])
            self.models[label] = model
        self.adapter_load = self.enterContext(patch.object(audit_pipeline.PeftModel, "from_pretrained", side_effect=self.load_adapter))
        self.encoded = []
        self.encode = self.enterContext(patch.object(audit_pipeline, "encode", side_effect=self.encode_record))
        self.loss = self.enterContext(patch.object(audit_pipeline, "loss_for", side_effect=self.checkpoint_loss))
        self.worker = self.enterContext(patch.object(audit_pipeline.subprocess, "run", side_effect=self.run_worker))

    @staticmethod
    def record(identifier):
        return {"id": identifier, "prompt": f"prompt {identifier}", "completion": "value = 1"}

    def load_adapter(self, base, checkpoint):
        return self.models["best600" if checkpoint.name == "checkpoint-600" else "stopped"]

    def encode_record(self, record, tokenizer, max_length, serialization):
        self.base_load.assert_not_called()
        self.assertIs(tokenizer, self.tokenizer)
        self.assertEqual((max_length, serialization), (2048, "raw"))
        self.assertNotIn("hf_target_loss", record)
        item = {"id": record["id"], "position": len(self.encoded)}
        self.encoded.append(item)
        return item

    def checkpoint_loss(self, model, item):
        self.assertIs(item, self.encoded[item["position"]])
        return (100 if model is self.models["best600"] else 200) + item["position"]

    def run_worker(self, command, **kwargs):
        if "--audit-output" in command:
            output = Path(command[command.index("--audit-output") + 1])
            output.mkdir()
            (output / "summary.json").write_text(json.dumps({"label": output.name}))
        return MagicMock(returncode=0)

    def run_audit(self, hf_samples=1, only=None):
        argv = ["--output", str(self.output), "--known-samples", "2", "--heldout-samples", "3", "--hf-samples", str(hf_samples)]
        if only:
            argv.extend(["--only", only])
        self.audit_pipeline.main(argv)

    def examples(self, label):
        return [json.loads(line) for line in (self.output / f"{label}-examples.jsonl").read_text().splitlines()]

    def test_encodes_each_sample_position_once_for_two_checkpoints(self):
        self.run_audit()
        self.assertEqual([item["id"] for item in self.encoded], ["known-0", "heldout-0"])
        self.assertEqual(self.encode.call_count, 2)
        self.assertEqual(self.loss.call_args_list, [call(model, item) for model in self.models.values() for item in self.encoded])
        self.assertEqual(self.base_load.call_count, 2)
        for label, value in (("best600", 100), ("stopped", 200)):
            examples = self.examples(label)
            self.assertEqual([record["id"] for record in examples], [record["id"] for record in self.known + self.heldout])
            self.assertEqual([record["audit_split"] for record in examples], ["known"] * 2 + ["heldout"] * 3)
            self.assertEqual([record.get("hf_target_loss") for record in examples], [value, None, value + 1, None, None])
            self.assertEqual(examples[0]["hf_prediction"]["source"], f"value = {value}\n")
            self.assertTrue(examples[0]["hf_prediction"]["end_marker_seen"])
            self.assertTrue(all("hf_prediction" not in record for record in examples[1:]))
            self.assertTrue(all(record["token_ids"] == [11, 12] for record in examples))
            self.assertEqual(self.models[label].generate.call_args.kwargs, {
                "input_ids": self.tokenizer.return_value.input_ids,
                "do_sample": False,
                "max_new_tokens": 192,
                "stop_strings": [self.audit_pipeline.END_CODE],
                "tokenizer": self.tokenizer,
            })
        self.assertEqual(json.loads((self.output / "summary.json").read_text()), {
            "best600": {"label": "best600"}, "stopped": {"label": "stopped"},
        })
        self.assertTrue(all("hf_target_loss" not in record for record in self.known + self.heldout))

    def test_overlapping_slices_keep_duplicate_positions_and_last_annotations(self):
        self.run_audit(hf_samples=3)
        self.assertEqual([item["id"] for item in self.encoded], ["known-0", "known-1", "heldout-0", "heldout-0", "heldout-1", "heldout-2"])
        self.assertEqual(self.encode.call_count, 6)
        self.assertEqual(self.loss.call_args_list, [call(model, item) for model in self.models.values() for item in self.encoded])
        for label, value in (("best600", 100), ("stopped", 200)):
            examples = self.examples(label)
            self.assertEqual([record["hf_target_loss"] for record in examples], [value, value + 1, value + 3, value + 4, value + 5])
            self.assertEqual(self.models[label].generate.call_count, 3)
            self.assertTrue(all("hf_prediction" in record for record in examples[:3]))
            self.assertTrue(all("hf_prediction" not in record for record in examples[3:]))

    def test_oversize_error_precedes_model_loading(self):
        self.encode.side_effect = ValueError("known-0 needs 2049 tokens, above 2048; refusing to truncate supervision")
        with self.assertRaisesRegex(ValueError, "refusing to truncate supervision"):
            self.run_audit()
        self.base_load.assert_not_called()
        self.adapter_load.assert_not_called()
        self.loss.assert_not_called()
        self.worker.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_only_selected_checkpoint_is_loaded_and_annotated(self):
        self.run_audit(only="stopped")
        self.assertEqual(self.encode.call_count, 2)
        self.assertEqual(self.loss.call_args_list, [call(self.models["stopped"], item) for item in self.encoded])
        self.base_load.assert_called_once()
        self.adapter_load.assert_called_once()
        self.assertEqual(self.adapter_load.call_args.args[1].name, "interrupted")
        self.models["best600"].generate.assert_not_called()
        self.assertFalse((self.output / "best600-examples.jsonl").exists())
        self.assertEqual([record.get("hf_target_loss") for record in self.examples("stopped")], [200, None, 201, None, None])
        self.assertEqual(json.loads((self.output / "summary.json").read_text()), {"stopped": {"label": "stopped"}})
