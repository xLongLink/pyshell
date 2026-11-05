"""The pinned llama.cpp adapter export command."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/base"
CONVERTER = ROOT / "artifacts/llama.cpp-b12038/convert_lora_to_gguf.py"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--base",
            str(BASE),
            "--outfile",
            str(args.output.resolve()),
            str(args.adapter.resolve()),
        ],
        check=False,
    ).returncode
