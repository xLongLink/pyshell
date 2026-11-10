import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from model import cache
from model.benchmark.src.settings import Settings


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    for name in (
        "model/benchmark/data/dev-tasks.jsonl",
        "model/benchmark/src/inference.py",
        "model/protocol/src/pyshell/serialization.py",
        "src/api/src/api/shell.py",
        "model/ci.py",
        "model/cache.py",
        ".github/workflows/model.yml",
        "model/train/data/train.jsonl",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    subprocess.run(["git", "add", "."], check=True)
    directory = tmp_path / "run"
    (directory / "baseline").mkdir(parents=True)
    (directory / "baseline/adapter_config.json").write_text(
        json.dumps({"revision": "a" * 40})
    )
    (directory / "baseline/adapter_model.safetensors").write_bytes(b"weights")
    (directory / "baseline.json").write_text(
        json.dumps({"repo": "test/model", "revision": "b" * 40})
    )
    check_output = subprocess.check_output

    def command(args, **kwargs):
        if args[0] == "docker":
            return "sha256:image\n"
        if args[0] == "nvidia-smi":
            return "RTX 4090, driver, 8.9\n"
        return check_output(args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", command)
    return directory


@pytest.mark.parametrize(
    "path",
    [
        "model/benchmark/data/dev-tasks.jsonl",
        "model/benchmark/src/inference.py",
        "model/protocol/src/pyshell/serialization.py",
        "src/api/src/api/shell.py",
        "model/ci.py",
        "model/cache.py",
        ".github/workflows/model.yml",
        "run/baseline/adapter_config.json",
        "run/baseline/adapter_model.safetensors",
        "run/baseline.json",
    ],
)
def test_evaluation_inputs_invalidate_cache(baseline, path):
    before = cache.fingerprint(baseline)
    target = baseline.parent / path
    if target.suffix == ".json":
        target.write_text(json.dumps({"repo": "test/model", "revision": "c" * 40}))
    else:
        target.write_bytes(target.read_bytes() + b"changed")
    assert cache.fingerprint(baseline) != before


def test_training_only_changes_keep_cache_key(baseline):
    before = cache.fingerprint(baseline)
    (baseline.parent / "model/train/data/train.jsonl").write_text("changed")
    assert cache.fingerprint(baseline) == before


def test_runtime_changes_invalidate_cache(baseline, monkeypatch):
    before = cache.fingerprint(baseline)
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "1")
    assert cache.fingerprint(baseline) != before
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE")
    monkeypatch.setattr(
        cache.importlib.metadata,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": "torch"}, version="different")],
    )
    assert cache.fingerprint(baseline) != before


@pytest.mark.parametrize("executable", ["docker", "nvidia-smi"])
def test_image_and_gpu_changes_invalidate_cache(baseline, monkeypatch, executable):
    before = cache.fingerprint(baseline)
    original = subprocess.check_output

    def command(args, **kwargs):
        return "changed" if args[0] == executable else original(args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", command)
    assert cache.fingerprint(baseline) != before


def test_key_is_stable_and_emitted_for_actions(baseline, monkeypatch):
    output = baseline / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(sys, "argv", ["cache", "key", "--directory", str(baseline)])
    cache.main()
    cache.main()
    first, second = output.read_text().splitlines()
    assert first == second
    assert first.startswith("key=baseline-v1-")
    (baseline / "baseline/adapter_model.safetensors").write_bytes(b"new weights")
    cache.main()
    assert output.read_text().splitlines()[-1] != first


@pytest.mark.parametrize("revision", [None, "main"])
def test_unpinned_base_model_cannot_be_cached(baseline, revision):
    (baseline / "baseline/adapter_config.json").write_text(
        json.dumps({"revision": revision})
    )
    with pytest.raises(ValueError, match="pinned"):
        cache.fingerprint(baseline)


def test_cache_roundtrip_validates_inputs_tasks_and_preserves_provenance(
    baseline, monkeypatch
):
    config = Settings()
    tasks = [
        json.loads(line)
        for line in (Path(config.data) / "dev-tasks.jsonl").read_text().splitlines()
    ]
    rows = [
        {"id": task["id"], "family": task["family"], "status": "passed"}
        for task in tasks
    ]
    output = baseline / "baseline.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    provenance = baseline / "baseline.jsonl.config.json"
    provenance.write_text('{"commit":"original"}')
    manifest = baseline / "baseline-cache-key.json"
    manifest.write_text('{"inputs":"original"}')

    def run(action):
        monkeypatch.setattr(
            sys, "argv", ["cache", action, "--directory", str(baseline)]
        )
        cache.main()

    run("store")
    output.unlink()
    provenance.unlink()
    manifest.write_text('{"inputs":"changed"}')
    with pytest.raises(ValueError, match="inputs do not match"):
        run("restore")
    assert not output.exists()
    manifest.write_text('{"inputs":"original"}')
    run("restore")
    assert [json.loads(line) for line in output.read_text().splitlines()] == rows
    assert json.loads(provenance.read_text()) == {"commit": "original"}
    with pytest.raises(FileExistsError):
        run("restore")
    output.unlink()
    (baseline.parent / ".baseline-cache/baseline.jsonl").write_text(
        json.dumps(rows[0]) + "\n"
    )
    with pytest.raises(ValueError, match="complete current benchmark"):
        run("restore")
    assert not output.exists()
