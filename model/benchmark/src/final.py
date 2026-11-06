import json

from model.benchmark.src import settings
from model.benchmark.src.artifact import load_artifact
from model.benchmark.src.corpus import load_tasks, resolve_data
from model.benchmark.src.evaluate import evaluate, inference_settings
from model.benchmark.src.proof_policy import development_ready
from model.benchmark.src.verification import verify


def run(config):
    output = config.output / "final"
    if output.exists():
        raise FileExistsError(f"Refusing to reopen an existing final evaluation: {output}")
    development = config.output / "candidate"
    summary = json.loads((development / "summary.json").read_text())
    if summary["split"] != "dev" or not development_ready(summary):
        raise ValueError("Development gate was not met; final holdout remains unopened")
    artifact = load_artifact(config.candidate, candidate=True)
    identity = {
        "adapter_sha256": artifact.weights_sha256,
        "adapter_config_sha256": artifact.config_sha256,
        "base_model": artifact.base,
        "base_revision": artifact.revision,
        "run_sha256": artifact.run_sha256,
        "inference": inference_settings(config),
    }
    if any(summary[key] != value for key, value in identity.items()):
        raise ValueError("Candidate identity or inference settings changed after development")
    data = resolve_data(config, "dev")
    _, digest, _ = load_tasks(data, "dev")
    if digest != summary["task_file_sha256"]:
        raise ValueError("Development tasks differ from the frozen manifest")
    data = resolve_data(config, "test")
    evaluate(artifact, config, output, data=data, split="test", final=True)
    verify(output, data, development)


def main():
    run(settings.SETTINGS)
