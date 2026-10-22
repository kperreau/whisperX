"""Exercise the real batching pipeline without downloading inference models."""

from types import SimpleNamespace

import numpy as np
import pytest

from whisperx import asr
from whisperx.diarize import Segment
from whisperx.vads.vad import Vad


class FakeVad(Vad):
    @staticmethod
    def preprocess_audio(audio):
        return audio

    def __call__(self, audio):
        return [Segment(0, 1), Segment(2, 3), Segment(4, 5)]


@pytest.fixture
def pipeline(monkeypatch):
    def generate(features, tokenizer, options):
        assert 7 in options.suppress_tokens
        return {"text": ["hello"] * len(features), "avg_logprob": [-0.5] * len(features)}

    model = SimpleNamespace(
        model=SimpleNamespace(is_multilingual=True),
        hf_tokenizer=None,
        feat_kwargs={"feature_size": 80},
        generate_segment_batched=generate,
    )
    monkeypatch.setattr(asr, "Tokenizer", lambda *args, task, language: SimpleNamespace(
        language_code=language, task=task))
    monkeypatch.setattr(asr, "find_numeral_symbol_tokens", lambda tokenizer: [7])
    result = asr.load_model("tiny", "cpu", model=model, vad_model=FakeVad(0.5),
                            asr_options={"suppress_numerals": True})
    monkeypatch.setattr(result, "detect_language", lambda audio: "en")
    return result


@pytest.mark.parametrize("batch_size", [None, 1, 2, 4])
def test_transcription_batches_keep_text_and_timestamps(pipeline, batch_size):
    options = pipeline.options
    progress = []
    result = pipeline.transcribe(np.zeros(5 * 16000, dtype=np.float32), batch_size=batch_size,
                                 chunk_size=1, progress_callback=progress.append)
    assert [segment["text"] for segment in result["segments"]] == ["hello"] * 3
    assert [segment["start"] for segment in result["segments"]] == [0, 2, 4]
    assert [segment["end"] for segment in result["segments"]] == [1, 3, 5]
    assert [segment["avg_logprob"] for segment in result["segments"]] == [-0.5] * 3
    assert result["language"] == "en"
    assert progress == pytest.approx([100 / 3, 200 / 3, 100])
    assert pipeline.tokenizer is None
    assert pipeline.options is options


@pytest.mark.parametrize("failure", ["inference", "callback"])
def test_transcription_restores_state_after_failure(pipeline, monkeypatch, failure):
    options = pipeline.options

    def fail(*args):
        raise RuntimeError("transcription interrupted")

    with monkeypatch.context() as patch:
        callback = None
        if failure == "inference":
            patch.setattr(pipeline.model, "generate_segment_batched", fail)
        else:
            callback = fail
        with pytest.raises(RuntimeError, match="transcription interrupted"):
            pipeline.transcribe(np.zeros(5 * 16000, dtype=np.float32), progress_callback=callback)

    assert pipeline.tokenizer is None
    assert pipeline.options is options
    result = pipeline.transcribe(np.zeros(5 * 16000, dtype=np.float32))
    assert result["segments"][0]["text"] == "hello"
