import json
from pathlib import Path

from transformers import TrainerCallback


def load_resume(path, metadata):
    if not (path / "trainer_state.json").is_file():
        raise ValueError(
            "--resume requires a Trainer checkpoint directory; "
            "use --initialize-adapter for older adapters"
        )
    saved = json.loads((path / "training.json").read_text())
    if saved != metadata:
        raise ValueError(
            "Resume dataset, base model or supervision configuration differs from the checkpoint"
        )
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        if not (path / name).is_file():
            raise ValueError(f"Incomplete Trainer checkpoint: missing {name}")
    state = json.loads((path / "trainer_state.json").read_text())
    return state["global_step"]


class Checkpoints(TrainerCallback):
    def __init__(self, metadata, stop_update):
        self.metadata = metadata
        self.stop_update = stop_update

    def write(self, path):
        (path / "training.json").write_text(json.dumps(self.metadata, indent=2) + "\n")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_update:
            control.should_training_stop = True
            control.should_save = True
            control.should_evaluate = args.eval_strategy != "no"

    def on_save(self, args, state, control, **kwargs):
        path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self.write(path)
