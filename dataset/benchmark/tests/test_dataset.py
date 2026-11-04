"""Check the expanded corpus without loading a model or contacting providers."""

import json
import pytest
from pathlib import Path


class TestDataset:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def setup_class_data(cls):
        cls.root = Path(__file__).resolve().parents[2] / "data"
        cls.train = [json.loads(line) for line in (cls.root / "train.jsonl").read_text().splitlines()]
        cls.validation = [json.loads(line) for line in (cls.root / "validation.jsonl").read_text().splitlines()]

    def test_counts_uniqueness_and_heldout_compositions(self):
        assert self.train
        assert self.validation
        train_prompts = {record["prompt"] for record in self.train}
        validation_prompts = {record["prompt"] for record in self.validation}
        assert len(train_prompts) == len(self.train)
        assert len(validation_prompts) == len(self.validation)
        assert not train_prompts & validation_prompts
        assert not {record["family"] for record in self.train} & {record["family"] for record in self.validation}

    def test_targets_are_python_and_prompts_have_no_chat_roles(self, subtests):
        for record in self.train + self.validation:
            with subtests.test(id=record["id"]):
                compile(record["completion"], "<candidate>", "exec")
                assert record["prompt"].endswith("<|code|>\n")
                assert "<|im_start|>" not in record["prompt"]
                assert "messages" not in record
