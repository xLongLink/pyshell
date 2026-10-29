import pyshell
from pyshell.cli import main
from pyshell.agent import Agent
from pyshell.shell import PythonShell


def test_public_exports():
    assert pyshell.Agent is Agent
    assert pyshell.PythonShell is PythonShell
    assert pyshell.main is main
    assert set(pyshell.__all__) == {"Agent", "PythonShell", "main"}
