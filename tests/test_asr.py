"""Exercise the real batching pipeline without downloading inference models."""

from types import SimpleNamespace, MethodType
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

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
    def generate(features, tokenizer, options, previous_batch_context_tokens=None):
        assert 7 in options.suppress_tokens
        return {"text": ["hello"] * len(features), "avg_logprob": [-0.5] * len(features),
                "tokens": [[1]] * len(features)}

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
@pytest.mark.parametrize("interleaved_context", [False, True])
def test_transcription_restores_state_after_failure(pipeline, monkeypatch, failure, interleaved_context):
    options = pipeline.options

    def fail(*args, **kwargs):
        raise RuntimeError("transcription interrupted")

    with monkeypatch.context() as patch:
        callback = None
        if failure == "inference":
            patch.setattr(pipeline.model, "generate_segment_batched", fail)
        else:
            callback = fail
        with pytest.raises(RuntimeError, match="transcription interrupted"):
            pipeline.transcribe(np.zeros(5 * 16000, dtype=np.float32), progress_callback=callback,
                                batch_size=2, interleaved_context=interleaved_context)

    assert pipeline.tokenizer is None
    assert pipeline.options is options
    assert pipeline.previous_batch_context_tokens == []
    assert pipeline._use_interleaved_context is False
    result = pipeline.transcribe(np.zeros(5 * 16000, dtype=np.float32))
    assert result["segments"][0]["text"] == "hello"


def test_interleaved_context_keeps_streams_and_redo_scores_together(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline.vad_model, "merge_chunks", lambda *args, **kwargs: [
        {"start": i, "end": i + 1} for i in range(7)])
    monkeypatch.setattr(pipeline, "preprocess", lambda audio: {
        "inputs": torch.tensor([audio["inputs"][0]])})
    calls = []

    def generate(features, tokenizer, options, previous_batch_context_tokens=None):
        ids = [int(row[0]) for row in features]
        calls.append((ids, deepcopy(previous_batch_context_tokens)))
        return {"text": [f"segment {i}, pass {len(calls)}" for i in ids],
                "avg_logprob": [-float(len(calls))] * len(ids),
                "tokens": [[i + 1] for i in ids]}

    monkeypatch.setattr(pipeline.model, "generate_segment_batched", generate)
    audio = np.repeat(np.arange(7, dtype=np.float32), 16000)
    progress = []
    result = pipeline.transcribe(audio, batch_size=3, interleaved_context=True,
                                 progress_callback=progress.append)
    assert calls == [
        ([0, 3, 5], [[], [], []]),
        ([1, 4, 6], [[1], [4], [6]]),
        ([2], [[1, 2]]),
        ([0, 3, 5], [[], [1, 2, 3], [4, 5]]),
    ]
    assert [s["start"] for s in result["segments"]] == list(range(7))
    for i in [0, 3, 5]:
        assert result["segments"][i]["text"] == f"segment {i}, pass 4"
        assert result["segments"][i]["avg_logprob"] == -4
    assert progress == sorted(progress)
    assert progress[-1] == 100
    assert pipeline.previous_batch_context_tokens == []


def test_interleaved_context_handles_no_speech(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline.vad_model, "merge_chunks", lambda *args, **kwargs: [])
    result = pipeline.transcribe(np.zeros(16000, dtype=np.float32), batch_size=2,
                                 interleaved_context=True)
    assert result == {"segments": [], "language": "en"}
    assert pipeline.previous_batch_context_tokens == []


@pytest.mark.parametrize("prompt_length", [224, 300])
def test_initial_prompt_can_exhaust_context_budget(pipeline, monkeypatch, prompt_length):
    pipeline.options = replace(pipeline.options, initial_prompt="prompt", suppress_tokens=[7])
    pipeline.tokenizer = SimpleNamespace(encode=lambda text: list(range(prompt_length)))
    pipeline._use_interleaved_context = True
    pipeline.previous_batch_context_tokens = [[10, 11]]
    pipeline.last_segment_tokens_per_stream = [[]]
    pipeline.stream_segment_indices = [0]
    pipeline._forward({"inputs": torch.zeros((1, 1))})
    assert pipeline.previous_batch_context_tokens == [[]]


def test_generation_groups_context_lengths_without_end_of_text_padding(pipeline):
    calls = []

    def generate(encoded, prompts, **kwargs):
        ids = [int(row[0]) for row in encoded]
        calls.append((ids, prompts))
        assert len({len(prompt) for prompt in prompts}) == 1
        assert all(100 not in prompt for prompt in prompts)
        return [SimpleNamespace(sequences_ids=[[i + 1, 100]], scores=[-1.0]) for i in ids]

    model = SimpleNamespace(max_length=448, encode=lambda features: features,
                            model=SimpleNamespace(generate=generate))
    model.get_prompt = MethodType(asr.WhisperModel.get_prompt, model)
    tokenizer = SimpleNamespace(
        sot_prev=101, eot=100, sot_sequence=[102, 103], no_timestamps=104,
        tokenizer=SimpleNamespace(decode_batch=lambda tokens: [str(t[0]) for t in tokens]),
    )
    result = asr.WhisperModel.generate_segment_batched(
        model, np.arange(5).reshape(-1, 1), tokenizer, pipeline.options,
        previous_batch_context_tokens=[[], [10], [20, 21], [11], []],
    )
    assert [ids for ids, prompts in calls] == [[0, 4], [1, 3], [2]]
    assert result["text"] == ["1", "2", "3", "4", "5"]
    assert result["tokens"] == [[1], [2], [3], [4], [5]]
