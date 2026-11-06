import json


def test_library_logs_training_and_validation_losses(training_run):
    output = training_run("logged")
    state = json.loads((output / "trainer_state.json").read_text())
    losses = [row for row in state["log_history"] if "loss" in row]
    assert [row["step"] for row in losses] == [1, 2, 3, 4]
    assert all(row["loss"] > 0 for row in losses)
    assert any("eval_loss" in row for row in state["log_history"])


def test_default_run_saves_only_the_final_adapter(training_run):
    output = training_run("final-only")
    assert not list(output.glob("checkpoint-*"))
    assert (output / "adapter_model.safetensors").is_file()
    assert (output / "adapter_config.json").is_file()
    assert not (output / "optimizer.pt").exists()
    assert not (output / "training.json").exists()
    assert not (output / "run.json").exists()
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 4
    assert any("eval_loss" in row and row["step"] == 4 for row in state["log_history"])


def test_native_step_budget_counts_optimizer_updates(training_run):
    output = training_run("updates", steps=5)
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 5
