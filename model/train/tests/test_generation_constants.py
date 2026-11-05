import json

import pytest

from pyshell.serialization import INITIALIZATION
from model.train.src.generation import expanded, legacy


def test_legacy_startup_uses_serialization_initialization():
    assert legacy.worker_context()[1] == {"role": "user", "content": INITIALIZATION}
    assert (
        INITIALIZATION
        == "Initialize the Python terminal, report the preloaded names, and display ui's docstring."
    )


def test_expanded_rejects_wrong_legacy_count_before_verification(tmp_path, monkeypatch):
    source = tmp_path / "model/train/data/chat/raw/train.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"id": "only-record"}) + "\n")
    monkeypatch.setattr(expanded, "REPO", tmp_path)
    with pytest.raises(AssertionError):
        expanded.main(["--output", str(tmp_path / "output")])
