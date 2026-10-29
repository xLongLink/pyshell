import pytest
from script import main, Agent
from unittest.mock import patch
from script import Settings


@pytest.fixture
def run_main(monkeypatch, remote_shell):
    def run(inputs, run_effect):
        settings = Settings(_env_file=None, openai_api_key="test-key")
        instructions = iter(inputs)
        monkeypatch.setattr("script.Settings", lambda: settings)
        monkeypatch.setattr("script.RemoteShell", lambda url: remote_shell)
        monkeypatch.setattr("sys.argv", ["script.py"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(instructions))

        # Exercise the real input loop without making API requests.
        with (
            patch("openai.OpenAI"),
            patch.object(Agent, "warm_up"),
            patch.object(Agent, "run", autospec=True, side_effect=run_effect) as agent_run,
        ):
            status = main()
        return status, agent_run

    return run


def test_interactive_instructions_share_imports(run_main):
    def execute(agent, instruction):
        if instruction == "calculate":
            agent.shell.execute("import math; value = math.sqrt(16)")
        else:
            assert agent.shell.execute("print(math.sqrt(16), value)")["stdout"] == "4.0 4.0\n"

    status, run = run_main(["calculate", "continue", "/exit"], execute)

    # Both instructions must run on the same agent instance.
    assert status == 0
    assert run.call_count == 2
    assert run.call_args_list[0].args[0] is run.call_args_list[1].args[0]


def test_interactive_reset_clears_imports(run_main, capsys):
    def execute(agent, instruction):
        if instruction == "import":
            agent.shell.execute("import math")
        else:
            assert "NameError" in agent.shell.execute("print(math)")["exception"]

    status, run = run_main(["import", "/reset", "check", "/exit"], execute)

    # Reset is handled locally rather than sent to the model.
    assert status == 0
    assert run.call_count == 2
    assert "Python namespace and conversation reset" in capsys.readouterr().out


def test_cli_rejects_command_line_instruction(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["script.py", "calculate"])
    with patch("script.Settings") as settings:
        with pytest.raises(SystemExit) as exc:
            main()

    # Argument validation must happen before loading API settings.
    assert exc.value.code == 2
    assert "unrecognized arguments: calculate" in capsys.readouterr().err
    settings.assert_not_called()
