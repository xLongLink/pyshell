import json


def test_library_logs_losses_and_saves_resumable_final_checkpoint(training_run):
    output = training_run("logged")
    state = json.loads((output / "trainer_state.json").read_text())
    losses = [row for row in state["log_history"] if "loss" in row]
    assert [row["step"] for row in losses] == [1, 2, 3, 4]
    assert all(row["loss"] > 0 for row in losses)
    assert any("eval_loss" in row for row in state["log_history"])
    checkpoint = output / "checkpoint-4"
    for name in (
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training.json",
    ):
        assert (checkpoint / name).is_file()
    assert not (output / "steps.jsonl").exists()


def test_pausing_keeps_original_schedule_and_saves_on_optimizer_boundary(training_run):
    output = training_run("paused", "--stop-after", "4")
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 2
    assert state["max_steps"] == 4
    assert (output / "checkpoint-2" / "training.json").is_file()


def test_nondivisible_budget_rounds_up_to_an_optimizer_update(training_run):
    output = training_run("rounded", "--steps", "5")
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 3


def test_validation_can_be_disabled(training_run):
    output = training_run("no-validation", "--eval-samples", "0")
    state = json.loads((output / "trainer_state.json").read_text())
    assert not any("eval_loss" in row for row in state["log_history"])
