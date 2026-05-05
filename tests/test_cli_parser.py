"""Smoke tests for the CLI argument parser.

Python 3.14 made argparse strict about validating `help=` strings as
percent-format specs at registration time, so an unescaped `%` in a help
text now raises `ValueError: badly formed help string` *just by importing
__main__ and calling cli()`. Building the parser is enough to surface it.
"""

import pytest


def test_cli_parser_builds_on_py314():
    """Importing __main__ and constructing the parser must not crash.

    Regresses the bug from --auto_hotwords help string containing a literal
    "%" that argparse 3.14 treated as a format spec.
    """
    import argparse
    from whisperx.__main__ import cli  # noqa: F401  (ensures module import works)

    # Re-implement the parser registration inline using the same module
    # imports — if any add_argument() call has a malformed help string, this
    # raises ValueError on Python 3.14.
    import importlib
    main_mod = importlib.import_module("whisperx.__main__")
    # Force the function body's argparse.add_argument calls to run by parsing
    # --help (which triggers help-string validation) with SystemExit caught.
    with pytest.raises(SystemExit):
        try:
            main_mod.cli.__wrapped__()  # type: ignore[attr-defined]
        except AttributeError:
            # Not wrapped — call indirectly via sys.argv.
            import sys
            old_argv = sys.argv
            sys.argv = ["whisperx", "--help"]
            try:
                main_mod.cli()
            finally:
                sys.argv = old_argv
