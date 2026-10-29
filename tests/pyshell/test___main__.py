import runpy
import pytest
from unittest.mock import Mock


def test_module_exits_with_cli_status(monkeypatch):
    main = Mock(return_value=7)
    monkeypatch.setattr("pyshell.cli.main", main)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("pyshell", run_name="__main__")

    assert exc.value.code == 7
    main.assert_called_once_with()
