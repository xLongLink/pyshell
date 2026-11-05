import json
from pathlib import Path

import pytest
import torch
from tokenizers import pre_tokenizers
from transformers import (
    AutoModelForImageTextToText,
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5Tokenizer,
    Qwen3_5VisionConfig,
)

from model.train.src import training


@pytest.fixture
def training_run(tmp_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}
    vocab.update(
        {
            char: i + 3
            for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
        }
    )
    tokenizer = Qwen3_5Tokenizer(
        vocab=vocab,
        merges=[],
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    text = Qwen3_5TextConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        pad_token_id=1,
        eos_token_id=2,
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=32,
    )
    config = Qwen3_5Config(text_config=text.to_dict(), vision_config=vision.to_dict())
    model = AutoModelForImageTextToText.from_config(config)
    base = tmp_path / "base"
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    data = tmp_path / "data"
    data.mkdir()
    rows = [
        {"id": str(i), "prompt": "prompt " * (i + 1), "completion": "code " * (i + 1)}
        for i in range(4)
    ]
    for split in ("train", "validation"):
        (data / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    def run(name, *extra):
        inputs, config = training.parse_args(
            [
                "--base",
                str(base),
                "--data",
                str(data),
                "--output_dir",
                str(tmp_path / name),
                "--use_cpu",
                "true",
                "--bf16",
                "false",
                "--per_device_train_batch_size",
                "1",
                "--per_device_eval_batch_size",
                "1",
                "--max_steps",
                "4",
                "--gradient_accumulation_steps",
                "2",
                "--save_steps",
                "2",
                "--logging_steps",
                "1",
                "--eval_strategy",
                "steps",
                "--eval_steps",
                "4",
                "--eval_on_start",
                "true",
                "--learning_rate",
                "0.001",
                "--optim",
                "adamw_torch",
                *extra,
            ]
        )
        training.train(inputs, config)
        return Path(config.output_dir)

    yield run
    torch.set_num_threads(threads)
