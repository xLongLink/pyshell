import hashlib
import json
from unittest.mock import patch

from model.benchmark.src.transfer_families import FAMILIES
from model.train.src.generation import transfer
from model.train.src.generation.binding_evaluation import evaluation_task


def test_binding_evaluation_preserves_complete_records_without_transfer_templates():
    with patch.object(transfer, "TEMPLATES", {}):
        records = [
            evaluation_task(family, pair, variant, seed, split)
            for family in FAMILIES
            for seed in (183731, 731971, 813797)
            for split in ("dev", "test")
            for pair in range(4)
            for variant in (0, 1)
        ]
    assert hashlib.sha256(json.dumps(records).encode()).hexdigest() == (
        "37b12b29ca33629e410269f23c67f7843b95fae89145419ed7e9017b0cbf8a9a"
    )
