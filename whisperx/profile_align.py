"""
Profiling helper for the alignment pipeline.

Splits the per-segment work into four buckets and reports cumulative timings:
  1. align_model forward (CTC emission computation, GPU/CPU)
  2. dp                  (trellis + backtrack + merge_repeats — Rust kernel)
  3. post                (timestamp reconstruction + sentence/word grouping
                          + interpolation — also inside the Rust kernel)
  4. speaker             (assign_word_speakers, diarization mapping)

Usage:

    from whisperx.profile_align import ProfiledAlign
    pa = ProfiledAlign(audio, align_model, metadata, device)
    result = pa.run(transcript)
    pa.report()

For deeper sampling profiling, run the script via:

    py-spy record -o flame.svg --rate 250 --subprocesses -- python -m whisperx.profile_align audio.wav
    scalene --cpu --gpu --profile-only whisperx/alignment.py -- python -m whisperx.profile_align audio.wav
"""
from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

import numpy as np
import torch

from whisperx.alignment import align as _align, load_align_model
from whisperx.audio import SAMPLE_RATE, load_audio


@dataclass
class _Bucket:
    name: str
    total: float = 0.0
    calls: int = 0

    def add(self, dt: float) -> None:
        self.total += dt
        self.calls += 1


@dataclass
class Profile:
    buckets: dict[str, _Bucket] = field(default_factory=dict)

    @contextmanager
    def time(self, name: str):
        b = self.buckets.setdefault(name, _Bucket(name))
        t0 = time.perf_counter()
        try:
            yield
        finally:
            b.add(time.perf_counter() - t0)

    def report(self) -> None:
        total = (self.buckets["total_align"].total + self.buckets.get("speaker.assign", _Bucket("speaker.assign")).total
                 if "total_align" in self.buckets else sum(b.total for b in self.buckets.values())) or 1e-9
        width = max((len(b.name) for b in self.buckets.values()), default=8)
        print(f"\n{'stage':<{width}}  {'calls':>6}  {'total(s)':>10}  {'%':>6}")
        print("-" * (width + 30))
        for b in sorted(self.buckets.values(), key=lambda x: -x.total):
            print(f"{b.name:<{width}}  {b.calls:>6}  {b.total:>10.3f}  "
                  f"{100 * b.total / total:>5.1f}%")
        print("-" * (width + 30))
        print(f"{'TOTAL':<{width}}  {'':>6}  {total:>10.3f}")


# Public API ────────────────────────────────────────────────────────────────


class ProfiledAlign:
    """
    Wraps `whisperx.alignment.align` with stage-level timers.

    Uses an instance-local emission engine timer for Torch and ONNX. The Rust
    entry point is temporarily patched, so use this profiler in an isolated
    process, not concurrently with other alignment calls. total_align includes
    model and Rust time; report percentages use that inclusive total once.
    """

    def __init__(self, audio, model, metadata, device: str = "cpu"):
        self.audio = audio
        self.model = model
        self.metadata = metadata
        self.device = device
        self.profile = Profile()

    def run(
        self,
        transcript: Iterable[dict],
        return_char_alignments: bool = False,
        diarize_df=None,
        **align_options,
    ):
        from whisperx import alignment as _alignment_module

        prof = self.profile

        from whisperx.alignment_engine import get_emission_engine
        backend = get_emission_engine(self.model, self.metadata)
        device = self.device

        class TimedEngine:
            stride = backend.stride
            receptive_field = backend.receptive_field

            def infer(self, *args):
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize(device)
                with prof.time("align_model.forward"):
                    out = backend.infer(*args)
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize(device)
                    return out

        # Patch: Rust kernel (covers "dp + post-processing")
        orig_rust = _alignment_module._rust_align_segment
        if orig_rust is None:
            raise RuntimeError("Rust extension whisperx_ext is not built")

        def timed_rust(*a, **kw):
            with prof.time("rust.dp+post"):
                return orig_rust(*a, **kw)

        _alignment_module._rust_align_segment = timed_rust
        try:
            with prof.time("total_align"):
                result = _align(
                    transcript=transcript,
                    model=TimedEngine(),
                    align_model_metadata={**self.metadata, "type": "custom"},
                    audio=self.audio,
                    device=self.device,
                    return_char_alignments=return_char_alignments,
                    **align_options,
                )
        finally:
            _alignment_module._rust_align_segment = orig_rust

        if diarize_df is not None:
            from whisperx.diarize import assign_word_speakers
            with prof.time("speaker.assign"):
                result = assign_word_speakers(diarize_df, result)

        return result

    def report(self) -> None:
        self.profile.report()


# Optional CLI ──────────────────────────────────────────────────────────────


def _main() -> None:
    parser = argparse.ArgumentParser(description="Profile the WhisperX alignment stages.")
    parser.add_argument("audio", help="Path to a 16 kHz mono audio file.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--align-model", default=None)
    parser.add_argument("--n-segments", type=int, default=20,
                        help="If transcript is synthesized, how many fake segments to align.")
    args = parser.parse_args()

    audio = load_audio(args.audio)
    duration_s = len(audio) / SAMPLE_RATE
    print(f"Audio duration: {duration_s:.2f}s")

    model, meta = load_align_model(args.language, args.device, model_name=args.align_model)

    # Synthetic transcript covering the file uniformly — for profiling only.
    seg = duration_s / max(args.n_segments, 1)
    transcript = [
        {"start": i * seg, "end": (i + 1) * seg, "text": "this is a profiling segment used for timing measurements"}
        for i in range(args.n_segments)
    ]

    pa = ProfiledAlign(audio, model, meta, device=args.device)
    pa.run(transcript)
    pa.report()


if __name__ == "__main__":
    _main()
