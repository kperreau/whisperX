"""Smoke tests for CLI parsing, including Python 3.14 help validation."""

import sys

import pytest

from whisperx.__main__ import cli


@pytest.mark.parametrize("flag, expected", [
    ("--help", "--auto_hotwords"),
    ("--version", "whisperx"),
    ("--python-version", "Python"),
])
def test_cli_information_flags(flag, expected, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["whisperx", flag])
    with pytest.raises(SystemExit) as error:
        cli()
    assert error.value.code == 0
    assert expected in capsys.readouterr().out
