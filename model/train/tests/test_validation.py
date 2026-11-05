import pytest
from datasets import Dataset

from model.train.src.supervision import encode
from model.train.src.training import validation_dataset


class Tokenizer:
    def __init__(self):
        self.texts = []

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        self.texts.append(text)
        return list(map(ord, text))


@pytest.fixture
def records():
    return [
        {"id": str(index), "prompt": f"prompt {index}", "completion": f"code {index}"}
        for index in range(10)
    ]


@pytest.mark.parametrize("limit", [1, 4, 10, 20])
def test_sampling_matches_original_order_and_only_encodes_selected(records, limit):
    tokenizer = Tokenizer()
    original_tokenizer = Tokenizer()
    expected = (
        Dataset.from_list([encode(record, original_tokenizer, 1024) for record in records])
        .shuffle(seed=713)
        .select(range(min(limit, len(records))))
    )
    selected = (
        Dataset.from_list(records)
        .shuffle(seed=713)
        .select(range(min(limit, len(records))))
    )

    actual = validation_dataset(records, tokenizer, 1024, limit)

    assert actual.to_list() == expected.to_list()
    assert tokenizer.texts[::2] == selected["prompt"]
    assert len(tokenizer.texts) == 2 * min(limit, len(records))
    repeated = validation_dataset(records, Tokenizer(), 1024, limit)
    assert repeated.to_list() == actual.to_list()
    if limit >= len(records):
        assert selected["id"] != [record["id"] for record in records]


@pytest.mark.parametrize("empty,limit", [(False, 0), (True, 0), (True, 4)])
def test_disabled_or_empty_validation_does_not_encode(empty, limit):
    tokenizer = Tokenizer()
    invalid = [{"id": "invalid", "prompt": [], "completion": "code"}]

    assert validation_dataset([] if empty else invalid, tokenizer, 1024, limit) is None
    assert tokenizer.texts == []


@pytest.mark.parametrize(
    "replacement,error,message",
    [
        ({"prompt": "x" * 2048}, ValueError, "refusing to truncate supervision"),
        ({"prompt": []}, TypeError, "requires raw prompt and completion"),
        ({"completion": None}, TypeError, "requires raw prompt and completion"),
        ({"prompt": ""}, ValueError, "nonempty prompt"),
    ],
)
@pytest.mark.parametrize("selected", [False, True])
def test_invalid_records_only_fail_when_selected(
    records, replacement, error, message, selected
):
    order = Dataset.from_list(records).shuffle(seed=713)["id"]
    invalid_id = order[0] if selected else order[-1]
    records[int(invalid_id)].update(replacement)
    tokenizer = Tokenizer()

    if selected:
        with pytest.raises(error, match=message):
            validation_dataset(records, tokenizer, 1024, 3)
    else:
        actual = validation_dataset(records, tokenizer, 1024, 3)
        assert len(actual) == 3
        assert tokenizer.texts[::2] == [f"prompt {index}" for index in order[:3]]
