#!/usr/bin/env python3
"""Validate actual ASR, ONNX VAD and native word alignment on CPU or CUDA.

The default fixture is the public JFK speech used by OpenAI Whisper's tests.
CUDA requests fail when no NVIDIA GPU is available; there is no CPU fallback.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
from time import perf_counter
from urllib.request import urlopen

import ctranslate2
import onnxruntime
import torch
import whisperx
from whisperx.alignment import _HAS_RUST_EXT
from whisperx.vads.silero import Silero


FIXTURE = "https://raw.githubusercontent.com/openai/whisper/31243bad24cc746f07d4c8bfdd2d974872cb1803/tests/jfk.flac"
FIXTURE_SHA256 = "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715"
SILERO_REVISION = "fba061dc5559f696e62171e9a0741782b0fdc23c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--language", default="en")
    parser.add_argument("--vad-method", choices=("silero", "pyannote"), default="silero")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not _HAS_RUST_EXT:
        raise RuntimeError("The Rust alignment extension must be loaded")
    if args.device == "cuda":
        if not torch.cuda.is_available() or ctranslate2.get_cuda_device_count() < 1:
            raise RuntimeError("CUDA validation requires a working NVIDIA GPU for torch AND CTranslate2")
        torch.cuda.synchronize()
        # Exercise an actual CUDA kernel before model loading, including cuBLAS.
        matrix = torch.eye(16, device="cuda")
        torch.testing.assert_close(matrix @ matrix, matrix)
    torch.set_num_threads(2)
    audio_path = args.audio
    if audio_path is None:
        audio_path = args.output.parent / "jfk.flac"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        if not audio_path.exists():
            with urlopen(FIXTURE, timeout=60) as response:
                audio_path.write_bytes(response.read())
        if hashlib.sha256(audio_path.read_bytes()).hexdigest() != FIXTURE_SHA256:
            raise RuntimeError("Unexpected audio fixture hash")
    audio = whisperx.load_audio(str(audio_path))
    duration = len(audio) / 16000
    vad = None
    if args.vad_method == "silero":
        vad = Silero.__new__(Silero)
        vad.vad_onset = 0.5
        vad.chunk_size = 5
        vad.vad_pipeline, utils = torch.hub.load(
            f"snakers4/silero-vad:{SILERO_REVISION}", "silero_vad",
            onnx=True, trust_repo=True, skip_validation=True)
        vad.get_speech_timestamps = utils[0]
    model = whisperx.load_model("tiny", args.device, language=args.language,
                               compute_type="float16" if args.device == "cuda" else "int8",
                               vad_model=vad, vad_method=args.vad_method,
                               vad_options={"chunk_size": 5}, threads=2)
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "device": args.device, "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
              "torch_cuda": torch.version.cuda, "rust_alignment": _HAS_RUST_EXT,
              "vad_method": args.vad_method,
              "onnx_providers": onnxruntime.get_available_providers(), "duration": duration,
              "versions": {p: version(p) for p in ("torch", "torchvision", "torchaudio", "torchcodec",
                           "ctranslate2", "onnxruntime", "transformers", "huggingface-hub", "numpy")},
              "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest()}
    for name, interleaved in (("standard", False), ("interleaved", True)):
        started = perf_counter()
        transcript = model.transcribe(audio, batch_size=2, chunk_size=5, interleaved_context=interleaved)
        segments = transcript["segments"]
        text = " ".join(s["text"] for s in segments)
        if not segments or len(text.split()) < 10:
            raise RuntimeError(f"Empty or incomplete {name} transcription: {text}")
        assert [s["start"] for s in segments] == sorted(s["start"] for s in segments)
        assert model.previous_batch_context_tokens == []
        if args.audio is None:
            normalized = " ".join(re.sub(r"[^a-z ]", "", text.lower()).split())
            assert "ask not what your country" in normalized, text
        result[name] = {"seconds": round(perf_counter() - started, 3), "text": text}
    started = perf_counter()
    alignment_model, metadata = whisperx.load_align_model(
        args.language, args.device, quantize=args.device == "cpu")
    aligned = whisperx.align(segments, alignment_model, metadata, audio, args.device,
                            batch_size=4, return_char_alignments=True)
    words = aligned["word_segments"]
    assert len(words) > 10
    assert all(s["words"] for s in aligned["segments"])
    for word in words:
        assert 0 <= word["start"] <= word["end"] <= duration + 0.05, word
    result["alignment"] = {"seconds": round(perf_counter() - started, 3), "words": len(words),
                           "timestamps": words}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Validated {args.device}: {duration:.2f}s audio, {len(words)} aligned words; {args.output}")


if __name__ == "__main__":
    main()
