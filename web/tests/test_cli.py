import pytest
from unittest.mock import patch
from web.src.script import Agent, Settings, main


@pytest.fixture
def run_main(monkeypatch, remote_shell):
    def run(inputs, run_effect):
        settings = Settings(_env_file=None, openai_api_key="test-key")
        monkeypatch.setattr("web.src.script.Settings", lambda: settings)
        monkeypatch.setattr("web.src.script.RemoteShell", lambda url: remote_shell)
        monkeypatch.setattr("sys.argv", ["script.py"])

        # Exercise the real input loop without making API requests.
        with (
            patch("builtins.input", side_effect=inputs),
            patch("web.src.script.OpenAIProvider"),
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

    status, run = run_main(["calculate", "continue", KeyboardInterrupt()], execute)

    # Both instructions must run on the same agent instance.
    assert status == 0
    assert run.call_count == 2
    assert run.call_args_list[0].args[0] is run.call_args_list[1].args[0]


def test_startup_and_exit_make_no_model_requests(monkeypatch):
    settings = Settings(_env_file=None, openai_api_key="test-key")
    monkeypatch.setattr("web.src.script.Settings", lambda: settings)
    monkeypatch.setattr("sys.argv", ["script.py"])
    with patch("builtins.input", side_effect=KeyboardInterrupt), patch("web.src.script.OpenAIProvider") as provider:
        assert main() == 0
    provider.return_value.client.responses.create.assert_not_called()


def test_interactive_reset_clears_imports(run_main, capsys):
    def execute(agent, instruction):
        if instruction == "import":
            agent.shell.execute("import math")
        else:
            assert "NameError" in agent.shell.execute("print(math)")["exception"]

    status, run = run_main(["import", "/reset", "check", KeyboardInterrupt()], execute)

    # Reset is handled locally rather than sent to the model.
    assert status == 0
    assert run.call_count == 2
    assert "Python namespace and conversation reset" in capsys.readouterr().out


def test_ctrl_c_during_instruction_exits(run_main, capsys):
    def interrupt(agent, instruction):
        raise KeyboardInterrupt

    status, run = run_main(["calculate"], interrupt)
    assert status == 0
    assert run.call_count == 1
    assert "may still be running on the server" in capsys.readouterr().out


def test_ctrl_c_during_reset_exits(run_main, remote_shell):
    with patch.object(remote_shell, "reset", side_effect=KeyboardInterrupt):
        status, run = run_main(["/reset"], None)
    assert status == 0
    run.assert_not_called()


def test_eof_exits_without_instruction(run_main):
    status, run = run_main([EOFError()], None)
    assert status == 0
    run.assert_not_called()


def test_exit_and_quit_are_not_local_commands(run_main):
    status, run = run_main(["/exit", "/quit", KeyboardInterrupt()], None)
    assert status == 0
    assert [call.args[1] for call in run.call_args_list] == ["/exit", "/quit"]


def test_cli_rejects_command_line_instruction(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["script.py", "calculate"])
    with patch("web.src.script.Settings") as settings:
        with pytest.raises(SystemExit) as exc:
            main()

    # Argument validation must happen before loading API settings.
    assert exc.value.code == 2
    assert "unrecognized arguments: calculate" in capsys.readouterr().err
    settings.assert_not_called()
