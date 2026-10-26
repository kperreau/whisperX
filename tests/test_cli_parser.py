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


def test_align_only_cli_passes_thread_limit_to_onnx(monkeypatch, tmp_path):
    from unittest.mock import Mock
    import numpy as np
    import whisperx.transcribe as transcription

    load = Mock(return_value=(object(), {'language': 'en', 'type': 'onnx'}))
    monkeypatch.setattr(transcription, 'load_align_model', load)
    monkeypatch.setattr(transcription, 'load_audio', lambda _: np.zeros(16000, np.float32))
    monkeypatch.setattr(transcription, 'align', Mock(return_value={'segments': [], 'word_segments': []}))
    monkeypatch.setattr(transcription, 'get_writer', Mock(return_value=Mock()))
    monkeypatch.setattr(transcription.torch, 'set_num_threads', Mock())
    monkeypatch.setattr(transcription.torch, 'set_num_interop_threads', Mock())
    monkeypatch.setattr(sys, 'argv', ['whisperx', 'fixture.wav', '--align_only', '--language', 'en',
        '--transcript_text', 'hello', '--threads', '2', '--align_engine', 'onnx',
        '--align_model', str(tmp_path/'bundle'), '--output_dir', str(tmp_path)])
    cli()
    assert load.call_args.kwargs['onnx_threads'] == 2
    assert load.call_args.kwargs['quantize'] is False
