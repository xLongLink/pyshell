from pyshell.shell import PythonShell


def test_shell_starts_without_imports():
    shell = PythonShell()

    assert set(shell.namespace) == {"__name__", "done"}
    assert callable(shell.namespace["done"])
    assert shell.result() == {
        "stdout": "", "stderr": "", "exception": None, "done": False,
    }


def test_multiline_blocks_and_expression_display(capsys):
    shell = PythonShell()
    for line in (
        "values = []", "for i in range(3):", "    values.append(i)", "", "values", "done()",
    ):
        shell.execute_line(line)

    assert shell.namespace["values"] == [0, 1, 2]
    assert shell.result() == {
        "stdout": "[0, 1, 2]\n", "stderr": "", "exception": None, "done": True,
    }
    assert capsys.readouterr().out == "[0, 1, 2]\n"
