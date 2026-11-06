import json


def test_default_run_logs_losses_and_saves_only_the_final_adapter(training_run):
    output = training_run("default")
    state = json.loads((output / "trainer_state.json").read_text())
    losses = [row for row in state["log_history"] if "loss" in row]
    assert [row["step"] for row in losses] == [1, 2, 3, 4]
    assert all(row["loss"] > 0 for row in losses)
    assert [row["step"] for row in state["log_history"] if "eval_loss" in row] == [0, 4]
    assert not list(output.glob("checkpoint-*"))
    assert (output / "adapter_model.safetensors").is_file()
    assert (output / "adapter_config.json").is_file()
    assert not (output / "optimizer.pt").exists()
    assert not (output / "training.json").exists()
    assert not (output / "run.json").exists()
    assert state["global_step"] == 4


def test_native_step_budget_counts_optimizer_updates(training_run):
    output = training_run("updates", steps=5)
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 5
