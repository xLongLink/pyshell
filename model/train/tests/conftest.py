import json

import pytest
import torch
from pyshell.serialization import Prompt, target_text
from tokenizers import pre_tokenizers
from transformers import (
    AutoModelForImageTextToText,
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5Tokenizer,
    Qwen3_5VisionConfig,
)

from model.train.src import training
from model.train.src.settings import Settings


@pytest.fixture
def tokenizer():
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}
    vocab.update(
        {
            char: i + 3
            for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
        }
    )
    vocab["ĊĊ"] = len(vocab)
    tokenizer = Qwen3_5Tokenizer(
        vocab=vocab,
        merges=[("Ċ", "Ċ")],
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    return tokenizer


@pytest.fixture
def training_run(tmp_path, tokenizer):
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
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
        config = Qwen3_5Config(
            text_config=text.to_dict(), vision_config=vision.to_dict()
        )
        model = AutoModelForImageTextToText.from_config(config)
        base = tmp_path / "base"
        model.save_pretrained(base)
        tokenizer.save_pretrained(base)
        data = tmp_path / "data"
        data.mkdir()
        rows = []
        for i in range(4):
            prompt = Prompt()
            prompt.append({"role": "user", "content": "prompt " * (i + 1)})
            rows.append(
                {
                    "id": str(i),
                    "prompt": prompt.render(),
                    "completion": target_text("code " * (i + 1)),
                }
            )
        for split in ("train", "validation"):
            (data / f"{split}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )

        def run(name, **changes):
            config = Settings(
                **{
                    "base": str(base),
                    "data": str(data),
                    "base_revision": None,
                    "output": tmp_path / name,
                    "steps": 4,
                    "max_length": 1024,
                    "gradient_accumulation_steps": 2,
                    "learning_rate": 0.001,
                    **changes,
                }
            )
            training.train(config)
            return config.output

        yield run
    finally:
        torch.set_num_threads(threads)
