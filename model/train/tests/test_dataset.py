"""Check fresh generated corpora without restoring previous training artifacts."""

import pytest

from model.benchmark.src.corpus import read_jsonl, write_jsonl
from model.train.src.generation.expanded import make_record


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("corpus")
    splits = {}
    for split, families, seed in (
        ("train", range(40), 862019),
        ("validation", range(40, 50), 429371),
    ):
        records = [
            make_record(variant * len(families) + index, family, seed, split, [])
            for variant in range(5)
            for index, family in enumerate(families)
        ]
        path = directory / f"{split}.jsonl"
        write_jsonl(path, records)
        splits[split] = read_jsonl(path)
    return splits


class TestDataset:
    def test_counts_uniqueness_and_heldout_compositions(self, corpus):
        train, validation = corpus["train"], corpus["validation"]
        assert len(train) == 200
        assert len(validation) == 50
        train_prompts = {record["prompt"] for record in train}
        validation_prompts = {record["prompt"] for record in validation}
        assert len(train_prompts) == len(train)
        assert len(validation_prompts) == len(validation)
        assert not train_prompts & validation_prompts
        assert not {record["family"] for record in train} & {
            record["family"] for record in validation
        }
        assert len({record["family"] for record in train}) == 40
        assert len({record["family"] for record in validation}) == 10

    def test_targets_are_python_and_prompts_have_no_chat_roles(self, corpus, subtests):
        for record in corpus["train"] + corpus["validation"]:
            with subtests.test(id=record["id"]):
                compile(record["completion"], "<candidate>", "exec")
                assert record["prompt"].endswith("<|code|>\n")
                assert "<|im_start|>" not in record["prompt"]
                assert "messages" not in record
