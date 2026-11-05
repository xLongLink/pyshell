import pytest
from pyshell.serialization import END_CODE
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from model.train.src.supervision import encode


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(char) for char in text]


def test_raw_training_masks_prompt_and_teaches_end_marker():
    record = {
        "id": "example",
        "prompt": "<|task|>\nSum values.\n<|code|>\n",
        "completion": "result = sum(values)",
    }
    item = encode(record, Tokenizer(), 1024)
    start = len(record["prompt"])
    assert item["input_ids"] == list(
        map(ord, record["prompt"] + record["completion"] + "\n" + END_CODE)
    )
    assert item["labels"] == [-100] * start + item["input_ids"][start:]


def test_library_collator_preserves_mask_and_ignores_padding():
    records = [
        {"id": "short", "prompt": "abc", "completion": "x"},
        {"id": "long", "prompt": "abcdef", "completion": "xyz"},
    ]
    items = [encode(record, Tokenizer(), 1024) for record in records]
    collator = DataCollatorForLanguageModeling(pad_token_id=0)
    batch = collator(items)
    for i, item in enumerate(items):
        length = len(item["input_ids"])
        assert batch["labels"][i, :length].tolist() == item["labels"]
        assert (batch["labels"][i, length:] == -100).all()


def test_overlength_examples_are_rejected_not_truncated():
    with pytest.raises(ValueError, match="refusing to truncate supervision"):
        encode({"id": "too-long", "prompt": "abc", "completion": "def"}, Tokenizer(), 3)


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"prompt": "abc"},
        {"completion": "def"},
        {"prompt": [], "completion": "def"},
        {"prompt": "abc", "completion": {}},
    ],
)
def test_raw_text_fields_are_required(fields):
    with pytest.raises(TypeError, match="requires raw prompt and completion"):
        encode({"id": "invalid", **fields}, Tokenizer(), 1024)


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError, match="nonempty prompt"):
        encode({"id": "empty", "prompt": "", "completion": "code"}, Tokenizer(), 1024)
