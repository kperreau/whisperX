"""Tests for torchaudio-optional alignment model loading."""

from unittest import mock
import builtins
import subprocess
import sys

import pytest

import whisperx.alignment as A


def test_default_torch_models_have_hf_fallback():
    """Every default torchaudio.pipelines align model should have an HF
    equivalent, so platforms without torchaudio wheels (e.g. Ascend /
    Linux aarch64) can still run forced alignment via Hugging Face."""
    assert set(A.DEFAULT_ALIGN_MODELS_TORCH.values()) == set(
        A.TORCHAUDIO_PIPELINE_TO_HF.keys()
    )


def test_hf_fallback_names_are_valid_checkpoints():
    """HF fallback names should be official facebook wav2vec2 checkpoints."""
    for name in A.TORCHAUDIO_PIPELINE_TO_HF.values():
        assert name.startswith("facebook/"), name


def test_load_align_model_falls_back_to_hf_when_torchaudio_missing():
    """With torchaudio absent, a torchaudio.pipelines default model name must
    be transparently remapped to its HF checkpoint and loaded via
    Wav2Vec2 (the HuggingFace path), not crash."""
    fake_vocab = {"<pad>": 0, "a": 1, "b": 2}
    fake_processor = mock.MagicMock()
    fake_processor.tokenizer.get_vocab.return_value = fake_vocab
    fake_model = mock.MagicMock()

    with mock.patch.object(A, "_load_torchaudio", return_value=None), mock.patch(
        "transformers.Wav2Vec2Processor"
    ) as proc_cls, mock.patch("transformers.Wav2Vec2ForCTC") as model_cls:
        proc_cls.from_pretrained.return_value = fake_processor
        model_cls.from_pretrained.return_value = fake_model

        model, metadata = A.load_align_model(
            language_code="en", device="cpu"
        )

    assert model is fake_model.to.return_value  # returned after .to(device)
    assert metadata["type"] == "huggingface"
    assert metadata["dictionary"] == {c.lower(): i for c, i in fake_vocab.items()}
    # the torchaudio bundle name was remapped to the HF checkpoint
    remapped_name = proc_cls.from_pretrained.call_args.args[0]
    assert remapped_name == A.TORCHAUDIO_PIPELINE_TO_HF["WAV2VEC2_ASR_BASE_960H"]
    assert model_cls.from_pretrained.call_args.args[0] == remapped_name


@pytest.mark.parametrize("error", [ModuleNotFoundError("torchaudio"), OSError("missing native library")])
def test_torchaudio_loader_handles_missing_and_broken_installs(error):
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "torchaudio":
            raise error
        return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=import_module):
        assert A._load_torchaudio() is None


def test_alignment_import_keeps_model_loaders_lazy():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import whisperx.alignment; "
        "assert 'torchaudio' not in sys.modules; "
        "assert 'transformers' not in sys.modules; "
        "assert 'numba' not in sys.modules",
    ], check=True, capture_output=True, text=True)
