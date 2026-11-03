"""The pinned llama.cpp adapter export command."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/base"
CONVERTER = ROOT / "artifacts/llama.cpp-b12038/convert_lora_to_gguf.py"


def export_command(adapter, output):
    return [
        sys.executable,
        str(CONVERTER),
        "--base",
        str(BASE),
        "--outfile",
        str(output),
        str(adapter),
    ]
