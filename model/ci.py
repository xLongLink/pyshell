import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def changes(paths: list[str]) -> tuple[bool, bool]:
    relevant = [
        path
        for path in paths
        if path.startswith("model/") or path == ".github/workflows/model.yml"
    ]
    relevant = [
        path for path in relevant if "/tests/" not in path and not path.endswith(".md")
    ]
    train = any(not path.startswith("model/benchmark/") for path in relevant)
    return bool(relevant), train


def score(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Evaluation must contain unique, nonempty tasks")
    statuses = {"passed", "incorrect", "call_limit", "boundary_error"}
    if any(row["status"] not in statuses for row in rows):
        raise ValueError("Unknown evaluation status")
    families = Counter(row["family"] for row in rows)
    passed = Counter(row["family"] for row in rows if row["status"] == "passed")
    return {
        "passed": sum(passed.values()),
        "total": len(rows),
        "success_rate": sum(passed.values()) / len(rows),
        "families": {
            family: {"passed": passed[family], "total": total}
            for family, total in sorted(families.items())
        },
    }


def compare(candidate: Path, baseline: Path) -> dict:
    candidate_rows = [json.loads(line) for line in candidate.read_text().splitlines()]
    baseline_rows = [json.loads(line) for line in baseline.read_text().splitlines()]
    candidate_score = score(candidate)
    baseline_score = score(baseline)
    if {(row["id"], row["family"]) for row in candidate_rows} != {
        (row["id"], row["family"]) for row in baseline_rows
    }:
        raise ValueError("Candidate and baseline must evaluate the same tasks")
    return {
        "candidate": candidate_score,
        "baseline": baseline_score,
        "improved": candidate_score["passed"] > baseline_score["passed"],
    }


def baseline(directory: Path, repo: str, bootstrap: bool) -> None:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    revision = None if bootstrap else api.model_info(repo).sha
    if revision is not None:
        snapshot_download(
            repo,
            revision=revision,
            local_dir=directory / "baseline",
            allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
        )
    (directory / "baseline.json").write_text(
        json.dumps({"repo": repo, "revision": revision})
    )


def train(directory: Path) -> None:
    from model.train.src import settings, training

    config = settings.Settings.model_validate(
        {
            **settings.SETTINGS.model_dump(),
            "cpu": False,
            "dtype": "bfloat16",
            "output": directory / "candidate",
        }
    )
    (directory / "training.json").write_text(config.model_dump_json(indent=2))
    training.train(config)


def evaluate(directory: Path, root: Path, adapter: str, output: str) -> None:
    sys.path.insert(0, str(root.resolve()))
    from model.benchmark.src import evaluate, settings

    config = settings.Settings.model_validate(
        {
            **settings.SETTINGS.model_dump(),
            "device": "cuda",
            "adapter": directory / adapter,
            "output": directory / output,
        }
    )
    (directory / f"{output}.config.json").write_text(
        json.dumps(
            {
                "commit": git("-C", str(root), "rev-parse", "HEAD"),
                "settings": config.model_dump(mode="json"),
            },
            indent=2,
        )
    )
    evaluate.run(config)


def report(directory: Path, bootstrap: bool, evaluation_only: bool = False) -> None:
    if evaluation_only:
        print(
            json.dumps(
                {
                    "approved": score(directory / "baseline.jsonl"),
                    "proposed": score(directory / "proposed.jsonl"),
                },
                indent=2,
            )
        )
        return
    result = {"candidate": score(directory / "candidate.jsonl"), "improved": True}
    if not bootstrap:
        result = compare(directory / "candidate.jsonl", directory / "baseline.jsonl")
    result.update(
        baseline_model=json.loads((directory / "baseline.json").read_text()),
        commit=git("rev-parse", "HEAD"),
        tree=git("rev-parse", "HEAD^{tree}"),
        bootstrap=bootstrap,
    )
    (directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = f"Candidate: {result['candidate']['success_rate']:.2%}\n"
    if not bootstrap:
        summary += f"Accepted: {result['baseline']['success_rate']:.2%}\n"
    summary += (
        "Bootstrap baseline\n" if bootstrap else f"Improved: {result['improved']}\n"
    )
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as file:
            file.write(summary)
    if not result["improved"]:
        raise SystemExit("Candidate did not strictly improve execution success")


def publish(directory: Path, repo: str) -> None:
    from huggingface_hub import HfApi

    metrics = json.loads((directory / "metrics.json").read_text())
    if not metrics["improved"] or metrics["tree"] != git("rev-parse", "HEAD^{tree}"):
        raise ValueError("Only the exact passing, merged tree can be published")
    if metrics["baseline_model"]["repo"] != repo:
        raise ValueError("Hugging Face repository changed since evaluation")
    api = HfApi()
    revision = metrics["baseline_model"]["revision"]
    if metrics["bootstrap"]:
        api.create_repo(repo, private=True, exist_ok=False)
        revision = api.model_info(repo).sha
    elif api.model_info(repo).sha != revision:
        raise ValueError("Accepted model changed; update the PR and evaluate again")
    for path in (directory / "candidate").glob("adapter_*"):
        shutil.copy2(path, directory / path.name)
    api.upload_folder(
        repo_id=repo,
        folder_path=directory,
        parent_commit=revision,
        allow_patterns=[
            "adapter_*",
            "metrics.json",
            "training.json",
            "baseline.json",
            "*.jsonl",
            "*.config.json",
        ],
        commit_message=f"Accept model from {metrics['commit']}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["changes", "baseline", "train", "evaluate", "report", "publish"],
    )
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--base")
    parser.add_argument("--repo", default=os.environ.get("HF_MODEL_REPO"))
    parser.add_argument("--adapter", default="candidate")
    parser.add_argument("--output", default="candidate.jsonl")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    args = parser.parse_args()
    if args.command == "changes":
        relevant, training = changes(
            git("diff", "--name-only", args.base, "HEAD").splitlines()
        )
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as file:
            file.write(
                f"relevant={str(relevant).lower()}\ntrain={str(training).lower()}\n"
            )
        return
    if args.directory is None:
        parser.error("--directory is required")
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.command in {"baseline", "publish"} and not args.repo:
        parser.error("Set the HF_MODEL_REPO repository variable")
    if args.command == "baseline":
        baseline(directory, args.repo, args.bootstrap)
    elif args.command == "train":
        sys.path.insert(0, str(Path.cwd()))
        train(directory)
    elif args.command == "evaluate":
        evaluate(directory, args.root, args.adapter, args.output)
    elif args.command == "report":
        report(directory, args.bootstrap, args.evaluation_only)
    elif args.command == "publish":
        publish(directory, args.repo)


if __name__ == "__main__":
    main()
