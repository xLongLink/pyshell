import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from model import ci
from model.benchmark.src.settings import Settings


def fingerprint(directory: Path) -> dict:
    native = json.loads((directory / "baseline/adapter_config.json").read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", native.get("revision") or ""):
        raise ValueError("Baseline caching requires a pinned base model revision")
    paths = ci.git(
        "ls-files",
        "model/benchmark",
        "model/protocol",
        "src",
        "model/ci.py",
        "model/cache.py",
        ".github/workflows/model.yml",
    ).splitlines()
    files = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths
    }
    for path in sorted((directory / "baseline").glob("adapter_*")):
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "files": files,
        "model": json.loads((directory / "baseline.json").read_text()),
        "python": platform.python_version(),
        "packages": sorted(
            (package.metadata["Name"], package.version)
            for package in importlib.metadata.distributions()
        ),
        "image": subprocess.check_output(
            [
                "docker",
                "image",
                "inspect",
                "pyshell-pyshell:latest",
                "--format",
                "{{.Id}}",
            ],
            text=True,
        ).strip(),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
                "OMP_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
    }


def validate(path: Path) -> None:
    ci.score(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    config = Settings()
    tasks = [
        json.loads(line)
        for line in (Path(config.data) / "dev-tasks.jsonl").read_text().splitlines()
    ]
    if {(row["id"], row["family"]) for row in rows} != {
        (task["id"], task["family"]) for task in tasks
    }:
        raise ValueError("Cached baseline must contain the complete current benchmark")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["key", "store", "restore"])
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    manifest = directory / "baseline-cache-key.json"
    cache = Path(".baseline-cache")
    if args.action == "key":
        inputs = fingerprint(directory)
        manifest.write_text(json.dumps(inputs, sort_keys=True))
        key = "baseline-v1-" + hashlib.sha256(manifest.read_bytes()).hexdigest()
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as file:
            file.write(f"key={key}\n")
    elif args.action == "store":
        validate(directory / "baseline.jsonl")
        cache.mkdir()
        shutil.copy2(manifest, cache / "manifest.json")
        for name in ("baseline.jsonl", "baseline.jsonl.config.json"):
            shutil.copy2(directory / name, cache / name)
    else:
        if (cache / "manifest.json").read_bytes() != manifest.read_bytes():
            raise ValueError("Cached baseline inputs do not match this run")
        validate(cache / "baseline.jsonl")
        for name in ("baseline.jsonl", "baseline.jsonl.config.json"):
            target = directory / name
            if target.exists():
                raise FileExistsError(target)
            shutil.copy2(cache / name, target)


if __name__ == "__main__":
    main()
