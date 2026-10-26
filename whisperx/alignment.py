"""
Forced Alignment with Whisper
C. Max Bain
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Union, List
from pathlib import Path
import platform

import re

import numpy as np
import torch
import warnings

from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.schema import (
    AlignedTranscriptionResult,
    SingleSegment,
    SingleAlignedSegment,
    SingleWordSegment,
    SegmentData,
    ProgressCallback,
)
from whisperx.log_utils import get_logger

try:
    from whisperx_ext.alignment import align_segment as _rust_align_segment
    _HAS_RUST_EXT = True
except ImportError:
    _rust_align_segment = None
    _HAS_RUST_EXT = False

logger = get_logger(__name__)

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.start()
        if end > start:
            spans.append((start, end))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans or [(0, len(text))]

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

TORCHAUDIO_PIPELINE_TO_HF = {
    "WAV2VEC2_ASR_BASE_960H": "facebook/wav2vec2-base-960h",
    "VOXPOPULI_ASR_BASE_10K_FR": "facebook/wav2vec2-base-10k-voxpopuli-fr",
    "VOXPOPULI_ASR_BASE_10K_DE": "facebook/wav2vec2-base-10k-voxpopuli-de",
    "VOXPOPULI_ASR_BASE_10K_ES": "facebook/wav2vec2-base-10k-voxpopuli-es",
    "VOXPOPULI_ASR_BASE_10K_IT": "facebook/wav2vec2-base-10k-voxpopuli-it",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": 'nguyenvulebinh/wav2vec2-base-vi-vlsp2020',
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
    "id": "cahya/wav2vec2-large-xlsr-indonesian",
}


def _patch_dynamic_linear_reduce_range(model: torch.nn.Module) -> None:
    import types
    try:
        DynLinear = torch.ao.nn.quantized.dynamic.Linear
    except AttributeError:
        return

    def _forward(self, x):
        return torch.ops.quantized.linear_dynamic(
            x, self._packed_params._packed_params, reduce_range=False
        )

    for module in model.modules():
        if isinstance(module, DynLinear):
            module.forward = types.MethodType(_forward, module)


def _load_torchaudio():
    # Keep native audio libraries lazy: alignment with an already loaded model
    # does not need them. Broken native installs can use the HF model loader.
    try:
        import torchaudio
        return torchaudio
    except Exception as error:
        logger.warning("Torchaudio unavailable, using Hugging Face alignment models: %s", error)
        return None


def load_align_model(
    language_code: str,
    device: str,
    model_name: Optional[str] = None,
    model_dir=None,
    model_cache_only: bool = False,
    quantize: bool = False,
    engine: str = "torch",
    onnx_threads: int = 0,
):
    if engine == "onnx":
        from whisperx.alignment_engine import OnnxEmissionEngine
        if device != "cpu":
            raise ValueError("ONNX alignment currently supports device='cpu' only")
        if model_name is None:
            raise ValueError("ONNX alignment requires model_name pointing to a local bundle")
        backend = OnnxEmissionEngine(model_name, threads=onnx_threads)
        if backend.metadata["language"] != language_code:
            raise ValueError("ONNX bundle language does not match requested language")
        return backend, backend.metadata
    if engine != "torch":
        raise ValueError(f"Unsupported alignment engine: {engine}")
    if model_name is None:
        # use default model
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            logger.error(f"No default alignment model for language: {language_code}. "
                         f"Please find a wav2vec2.0 model finetuned on this language at https://huggingface.co/models, "
                         f"then pass the model name via --align_model [MODEL_NAME]")
            raise ValueError(f"No default align-model for language: {language_code}")

    torchaudio = _load_torchaudio()

    if torchaudio is not None and model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        if model_cache_only:
            checkpoint_dir = Path(model_dir) if model_dir else Path(torch.hub.get_dir()) / "checkpoints"
            checkpoint = checkpoint_dir / Path(bundle._path).name
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Alignment checkpoint is not cached: {checkpoint}")
        align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        if torchaudio is None:
            model_name = TORCHAUDIO_PIPELINE_TO_HF.get(model_name, model_name)

        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=model_dir, local_files_only=model_cache_only)
            align_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=model_dir, local_files_only=model_cache_only)
        except Exception as e:
            print(e)
            print(f"Error loading model from huggingface, check https://huggingface.co/models for finetuned wav2vec2.0 models")
            raise ValueError(f'The chosen align_model "{model_name}" could not be found in huggingface (https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)')
        pipeline_type = "huggingface"
        align_model = align_model.to(device)
        labels = processor.tokenizer.get_vocab()
        align_dictionary = {char.lower(): code for char,code in processor.tokenizer.get_vocab().items()}

    align_model.eval()

    if quantize:
        if device != "cpu":
            logger.warning(
                "align_quantize is a CPU-only optimization (dynamic int8); "
                f"requested device is {device!r}, skipping quantization."
            )
        else:
            try:
                # Pick a working quantized engine. On macOS ARM the default may
                # be "none" (NoQEngine), causing quantize_dynamic to fail at the
                # first forward with `Didn't find engine for operation
                # quantized::linear_prepack`. Prefer the architecture-specific
                # engine, including x86/fbgemm before qnnpack on x86.
                supported = list(torch.backends.quantized.supported_engines)
                order = ("x86", "fbgemm", "qnnpack") if platform.machine().lower() in {"x86_64", "amd64"} else ("qnnpack", "x86", "fbgemm")
                preferred = [e for e in order if e in supported]
                if preferred and torch.backends.quantized.engine not in preferred:
                    torch.backends.quantized.engine = preferred[0]
                    logger.info(f"Set torch.backends.quantized.engine = {preferred[0]!r}")
                align_model.eval()
                align_model = torch.ao.quantization.quantize_dynamic(
                    align_model, {torch.nn.Linear}, dtype=torch.qint8,
                )
                if torch.backends.quantized.engine == "qnnpack":
                    # qnnpack ignores reduce_range=True (the hardcoded default
                    # in torch.ao.nn.quantized.dynamic.Linear) and emits a
                    # warning on every forward. Patch the forward to pass
                    # reduce_range=False explicitly.
                    _patch_dynamic_linear_reduce_range(align_model)
                logger.info("Applied dynamic int8 quantization to align model (Linear layers)")
            except Exception as e:
                logger.warning(f"Failed to apply dynamic quantization to align model: {e}")

    align_metadata = {"language": language_code, "dictionary": align_dictionary,
                      "type": pipeline_type, "model_name": str(model_name)}

    return align_model, align_metadata


def align(
    transcript: Iterable[SingleSegment],
    model: object,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
    progress_callback: ProgressCallback = None,
    batch_size: int = 8,
    chunk_seconds: Optional[float] = 10.0,
    context_seconds: float = 1.0,
    max_batch_seconds: float = 15.0,
    return_diagnostics: bool = False,
    unsupported_policy: str = "wildcard",
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
    """
    from whisperx.alignment_engine import get_emission_engine, infer_emissions
    from whisperx.alignment_text import prepare_alignment_text

    transcript = list(transcript)
    if unsupported_policy not in {"wildcard", "error"}:
        raise ValueError("unsupported_policy must be wildcard or error")

    if interpolate_method not in {"nearest", "ignore"}:
        warnings.warn(
            f"interpolate_method='{interpolate_method}' is deprecated; use 'nearest' or 'ignore'. "
            "Falling back to 'nearest'.",
            DeprecationWarning,
            stacklevel=2,
        )

    if _rust_align_segment is None:
        raise ImportError(
            "whisperx_ext Rust extension is not built. Run `maturin develop --release` "
            "from the project root, or `pip install .` to compile the extension."
        )

    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    no_spaces = model_lang in LANGUAGES_WITHOUT_SPACES

    # 1. Expand caller-supplied spoken spans and retain display indices.
    total_segments = len(transcript)
    # Store temporary processing values
    segment_data: dict[int, SegmentData] = {}
    diagnostics = []
    for sdx, segment in enumerate(transcript):
        # strip spaces at beginning / end, but keep track of the amount.
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")

        text = segment["text"]
        clean_char, clean_cdx, diagnostic = prepare_alignment_text(
            text, model_dictionary, no_spaces, segment.get("spoken_replacements", []),
        )
        diagnostic["segment_index"] = sdx
        diagnostic["status"] = "pending"
        if unsupported_policy == "error" and diagnostic["unsupported_characters"]:
            raise ValueError(f"Unsupported alignment characters in segment {sdx}: {diagnostic['unsupported_characters']}")
        diagnostics.append(diagnostic)
        per_word = text if no_spaces else text.split(" ")
        clean_wdx = list(range(len(per_word)))

        sentence_spans = _sentence_spans(text)

        segment_data[sdx] = {
            "clean_char": clean_char,
            "clean_cdx": clean_cdx,
            "clean_wdx": clean_wdx,
            "sentence_spans": sentence_spans
        }

    def _optional_fields(d: dict, seg) -> dict:
        """Add start/end/score to dict only when not NaN."""
        for key in ("start", "end", "score"):
            val = getattr(seg, key)
            if not np.isnan(val):
                d[key] = val
        return d

    # Resolve blank_id once.
    blank_id = 0
    for char, code in model_dictionary.items():
        if char == '[pad]' or char == '<pad>':
            blank_id = code

    # 2a. Pre-collect per-segment alignment inputs (waveform slice + eligibility),
    #     so we can batch the wav2vec2 forward pass below.
    @dataclass
    class _SegmentInput:
        sdx: int
        t1: float
        t2: float
        text: str
        avg_logprob: Optional[float]
        eligible: bool
        skip_reason: Optional[str]
        fallback: SingleAlignedSegment
        waveform: Optional[torch.Tensor]  # shape (1, original samples)
        content_length: int  # actual content length before engine padding

    segment_inputs: List[_SegmentInput] = []
    for sdx, segment in enumerate(transcript):
        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]
        avg_logprob = segment.get("avg_logprob")

        fallback: SingleAlignedSegment = {
            "start": t1,
            "end": t2,
            "text": text,
            "words": [],
            "chars": None,
        }
        if avg_logprob is not None:
            fallback["avg_logprob"] = avg_logprob
        if return_char_alignments:
            fallback["chars"] = []

        # Filter ineligible segments without sending audio through the model.
        if len(segment_data[sdx]["clean_char"]) == 0:
            segment_inputs.append(_SegmentInput(
                sdx=sdx, t1=t1, t2=t2, text=text, avg_logprob=avg_logprob,
                eligible=False,
                skip_reason="no characters in this segment found in model dictionary",
                fallback=fallback, waveform=None, content_length=0,
            ))
            continue
        if not (np.isfinite(t1) and np.isfinite(t2) and 0 <= t1 < t2):
            raise ValueError(f"Invalid alignment interval: {t1}, {t2}")
        if t1 >= MAX_DURATION:
            segment_inputs.append(_SegmentInput(
                sdx=sdx, t1=t1, t2=t2, text=text, avg_logprob=avg_logprob,
                eligible=False,
                skip_reason="original start time longer than audio duration",
                fallback=fallback, waveform=None, content_length=0,
            ))
            continue

        t2 = min(t2, MAX_DURATION)
        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)
        waveform_segment = audio[:, f1:f2]
        content_length = waveform_segment.shape[-1]
        if content_length == 0:
            raise ValueError("Alignment interval contains no audio samples")
        segment_inputs.append(_SegmentInput(
            sdx=sdx, t1=t1, t2=t2, text=text, avg_logprob=avg_logprob,
            eligible=True, skip_reason=None, fallback=fallback,
            waveform=waveform_segment, content_length=content_length,
        ))

    # 2b. Batched wav2vec2 forward — emissions cached per segment index.
    eligible_indices = [i for i, item in enumerate(segment_inputs) if item.eligible]
    emissions = infer_emissions(
        [segment_inputs[i].waveform[0, :segment_inputs[i].content_length] for i in eligible_indices],
        get_emission_engine(model, align_model_metadata), device,
        batch_size=max(1, int(batch_size)), chunk_seconds=chunk_seconds,
        context_seconds=context_seconds, max_batch_seconds=max_batch_seconds,
    )
    emissions_by_idx = dict(zip(eligible_indices, emissions))

    # 2c. Per-segment Rust DP + post-processing, in original transcript order.
    aligned_segments: List[SingleAlignedSegment] = []
    for item_idx, item in enumerate(segment_inputs):
        sdx = item.sdx

        if not item.eligible:
            logger.warning(
                f'Failed to align segment ("{item.text}"): {item.skip_reason}, '
                'resorting to original'
            )
            diagnostics[sdx]["status"] = "fallback"
            diagnostics[sdx]["reason"] = item.skip_reason
            aligned_segments.append(item.fallback)
            continue

        text_clean = "".join(segment_data[sdx]["clean_char"])
        duration = item.t2 - item.t1
        emission_np = emissions_by_idx[item_idx]

        rust_subsegments = _rust_align_segment(
            emission=emission_np,
            text=item.text,
            text_clean=text_clean,
            model_dictionary=model_dictionary,
            clean_cdx=segment_data[sdx]["clean_cdx"],
            sentence_spans=segment_data[sdx]["sentence_spans"],
            blank_id=blank_id,
            t1=item.t1,
            duration=duration,
            no_spaces=no_spaces,
            return_char_alignments=return_char_alignments,
            interpolate_words=interpolate_method != "ignore",
        )

        if rust_subsegments is None:
            diagnostics[sdx]["status"] = "fallback"
            diagnostics[sdx]["reason"] = "backtrack_failed"
            logger.warning(
                f'Failed to align segment ("{item.text}"): backtrack failed, '
                'resorting to original'
            )
            aligned_segments.append(item.fallback)
            continue

        diagnostics[sdx]["status"] = "approximate" if diagnostics[sdx]["unsupported_characters"] else "aligned"
        diagnostics[sdx]["frames"] = len(emission_np)
        diagnostics[sdx]["context_limited"] = (
            chunk_seconds is not None and duration > chunk_seconds + 2 * context_seconds
        )
        scores = []
        raw_words = [word for subsegment in rust_subsegments for word in subsegment.words]
        for diagnostic, word in zip(diagnostics[sdx]["words"], raw_words):
            if np.isfinite(word.score):
                diagnostic["score"] = word.score
                scores.append(word.score)
            else:
                diagnostic["method"] = "interpolated" if np.isfinite(word.start) else "unaligned"
                diagnostics[sdx]["status"] = "approximate"
        if scores:
            diagnostics[sdx]["mean_word_score"] = float(np.mean(scores))
            diagnostics[sdx]["min_word_score"] = min(scores)
        for s in rust_subsegments:
            sub: SingleAlignedSegment = {
                "text": s.text,
                "start": s.start,
                "end": s.end,
                "words": [_optional_fields({"word": w.word}, w) for w in s.words],
            }
            if s.chars is not None:
                sub["chars"] = [_optional_fields({"char": c.char}, c) for c in s.chars]
            if item.avg_logprob is not None:
                sub["avg_logprob"] = item.avg_logprob
            aligned_segments.append(sub)

        if progress_callback is not None:
            progress_callback(((sdx + 1) / total_segments) * 100)

    # create word_segments list
    word_segments: List[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments += segment["words"]

    result = {"segments": aligned_segments, "word_segments": word_segments}
    if return_diagnostics:
        result["diagnostics"] = diagnostics
    return result
