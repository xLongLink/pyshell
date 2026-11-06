import pytest
from pyshell.serialization import END_CODE, serialize, target_text
from transformers import AutoTokenizer

from model.train.src import training
from model.train.src.data import prepare


@pytest.fixture
def tokenizer(training_run, tmp_path):
    return AutoTokenizer.from_pretrained(tmp_path / "base")


@pytest.mark.parametrize("framing", ["raw", "boundary", "eos"])
def test_ready_completions_are_idempotent_and_use_existing_eos(tokenizer, framing):
    prompt = serialize([{"role": "user", "content": "Calculate."}])
    code = "result = 1"
    completion = code if framing == "raw" else target_text(code)
    if framing == "eos":
        completion += tokenizer.eos_token
    vocab = tokenizer.get_vocab()
    record = prepare({"prompt": prompt, "completion": completion}, tokenizer, 1024)
    assert record == {
        "prompt": prompt,
        "completion": target_text(code) + tokenizer.eos_token,
    }
    assert prepare(record, tokenizer, 1024) == record
    assert tokenizer.get_vocab() == vocab


def test_trl_owns_tokenization_masking_and_padding(training_run, monkeypatch):
    trainers = []
    trainer_class = training.SFTTrainer

    def capture(**kwargs):
        trainer = trainer_class(**kwargs)
        trainers.append(trainer)
        return trainer

    monkeypatch.setattr(training, "SFTTrainer", capture)
    training_run("native")
    trainer = trainers[0]
    tokenizer = trainer.processing_class
    rows = [trainer.train_dataset[i] for i in range(2)]
    batch = trainer.data_collator(rows)
    for index, row in enumerate(rows):
        prompt_ids = tokenizer(row["prompt"])["input_ids"]
        target_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
        assert row["completion"].endswith("\n" + END_CODE + tokenizer.eos_token)
        assert row["input_ids"] == prompt_ids + target_ids
        assert row["labels"] == [-100] * len(prompt_ids) + target_ids
        assert row["input_ids"][-1] == tokenizer.eos_token_id
        length = len(row["input_ids"])
        assert batch["labels"][index, :length].tolist() == row["labels"]
        assert (batch["labels"][index, length:] == -100).all()


def test_length_validation_counts_eos_and_never_truncates(tokenizer):
    record = {"prompt": "prompt", "completion": "code"}
    ready = prepare(record, tokenizer, None)
    length = len(tokenizer(ready["prompt"] + ready["completion"])["input_ids"])
    assert prepare(record, tokenizer, length) == ready
    with pytest.raises(ValueError, match="refusing to truncate"):
        prepare(record, tokenizer, length - 1)


def test_bpe_merges_across_prompt_boundary_are_rejected(tokenizer):
    record = {"prompt": "prompt\n", "completion": "\ncode"}
    assert (
        tokenizer("\n")["input_ids"] + tokenizer("\n")["input_ids"]
        != tokenizer("\n\n")["input_ids"]
    )
    with pytest.raises(ValueError, match="token boundary"):
        prepare(record, tokenizer, 1024)


@pytest.mark.parametrize("field,value", [("prompt", []), ("completion", None)])
def test_training_fields_must_be_plain_text(tokenizer, field, value):
    with pytest.raises(TypeError, match="prompt and completion strings"):
        prepare(
            {"prompt": "prompt", "completion": "code", field: value}, tokenizer, 1024
        )


def test_empty_prompt_is_rejected(tokenizer):
    with pytest.raises(ValueError, match="nonempty prompt"):
        prepare({"prompt": "", "completion": "code"}, tokenizer, 1024)


def test_reserved_markers_inside_code_are_rejected(tokenizer):
    with pytest.raises(ValueError, match="reserved framing marker"):
        prepare({"prompt": "prompt", "completion": "x = '<|task|>'"}, tokenizer, 1024)
