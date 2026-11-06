import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from model.benchmark.src.corpus import adapter_weights, sha256


class Artifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Model identity
    base: str = Field(min_length=1)
    path: Path
    revision: str | None = Field(pattern=r"^[0-9a-f]{40}$")
    run_sha256: str | None = None
    config_sha256: str
    weights_sha256: str

    def verify(self):
        for filename, digest in (
            ("adapter_model.safetensors", self.weights_sha256),
            ("adapter_config.json", self.config_sha256),
            ("run.json", self.run_sha256),
        ):
            if digest is not None and sha256(self.path / filename) != digest:
                raise RuntimeError(f"Artifact changed since its identity was loaded: {filename}")


def load_artifact(path, *, candidate=False):
    native = json.loads((path / "adapter_config.json").read_text())
    training = path / "training.json"
    run = path / "run.json"
    if candidate and (not training.is_file() or not run.is_file()):
        raise ValueError("Candidate requires training.json and run.json provenance")
    identity = (
        json.loads(training.read_text())
        if training.is_file()
        else {"base_model": native["base_model_name_or_path"], "base_revision": native.get("revision")}
    )
    base = identity["base_model"]
    revision = identity["base_revision"]
    if not Path(base).is_dir() and revision is None:
        raise ValueError("Artifact requires a pinned base revision or an available local base model")
    canonical = str(Path(base).resolve()) if Path(base).is_dir() else base
    native_base = native["base_model_name_or_path"]
    native_base = str(Path(native_base).resolve()) if Path(native_base).is_dir() else native_base
    if native_base != canonical or native.get("revision") not in (None, revision):
        raise ValueError("Adapter configuration and training identity disagree")
    if run.is_file():
        selected = json.loads(run.read_text())["settings"]
        selected_base = str(Path(selected["base"]).resolve()) if Path(selected["base"]).is_dir() else selected["base"]
        if selected_base != canonical or selected["base_revision"] != revision:
            raise ValueError("run.json and training.json model identity disagree")
    return Artifact(
        base=canonical,
        path=path.resolve(),
        revision=revision,
        run_sha256=sha256(run) if run.is_file() else None,
        config_sha256=sha256(path / "adapter_config.json"),
        weights_sha256=sha256(adapter_weights(path)),
    )
