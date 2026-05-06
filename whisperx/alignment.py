"""
Forced Alignment with Whisper
C. Max Bain
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Union, List

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import warnings

from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.utils import PUNKT_LANGUAGES
from whisperx.schema import (
    AlignedTranscriptionResult,
    SingleSegment,
    SingleAlignedSegment,
    SingleWordSegment,
    SegmentData,
    ProgressCallback,
)
import nltk
from nltk.data import load as nltk_load
from whisperx.log_utils import get_logger

try:
    from whisperx_ext.alignment import align_segment as _rust_align_segment
    _HAS_RUST_EXT = True
except ImportError:
    _rust_align_segment = None
    _HAS_RUST_EXT = False

logger = get_logger(__name__)

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
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


def load_align_model(
    language_code: str,
    device: str,
    model_name: Optional[str] = None,
    model_dir=None,
    model_cache_only: bool = False,
    quantize: bool = False,
):
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

    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
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

    if quantize:
        if device != "cpu":
            logger.warning(
                "align_quantize is a CPU-only optimization (dynamic int8); "
                f"requested device is {device!r}, skipping quantization."
            )
        else:
            try:
                align_model.eval()
                align_model = torch.ao.quantization.quantize_dynamic(
                    align_model, {torch.nn.Linear}, dtype=torch.qint8,
                )
                logger.info("Applied dynamic int8 quantization to align model (Linear layers)")
            except Exception as e:
                logger.warning(f"Failed to apply dynamic quantization to align model: {e}")

    align_metadata = {"language": language_code, "dictionary": align_dictionary, "type": pipeline_type}

    return align_model, align_metadata


def align(
    transcript: Iterable[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
    progress_callback: ProgressCallback = None,
    batch_size: int = 8,
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
    """
    if interpolate_method != "nearest":
        warnings.warn(
            f"interpolate_method='{interpolate_method}' is deprecated, only 'nearest' is supported. "
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
    model_type = align_model_metadata["type"]
    no_spaces = model_lang in LANGUAGES_WITHOUT_SPACES

    # 1. Preprocess to keep only characters in dictionary
    total_segments = len(transcript)
    # Store temporary processing values
    segment_data: dict[int, SegmentData] = {}
    for sdx, segment in enumerate(transcript):
        # strip spaces at beginning / end, but keep track of the amount.
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")

        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # split into words
        if not no_spaces:
            per_word = text.split(" ")
        else:
            per_word = text

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.lower()
            # wav2vec2 models use "|" character to represent spaces
            if not no_spaces:
                char_ = char_.replace(" ", "|")

            # ignore whitespace at beginning and end of transcript
            if cdx < num_leading:
                pass
            elif cdx > len(text) - num_trailing - 1:
                pass
            elif char_ in model_dictionary.keys():
                clean_char.append(char_)
                clean_cdx.append(cdx)
            elif char_ not in (" ", "|"):
                # unknown char (digit, symbol, foreign script) — use wildcard
                clean_char.append(char_)
                clean_cdx.append(cdx)

        clean_wdx = list(range(len(per_word)))

        # Use language-specific Punkt model if available otherwise we fallback to English.
        punkt_lang = PUNKT_LANGUAGES.get(model_lang, 'english')
        try:
            sentence_splitter = nltk_load(f'tokenizers/punkt_tab/{punkt_lang}.pickle')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
            sentence_splitter = nltk_load(f'tokenizers/punkt_tab/{punkt_lang}.pickle')
        sentence_spans = list(sentence_splitter.span_tokenize(text))

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
        waveform: Optional[torch.Tensor]  # shape (1, T_orig_or_padded_to_400)
        content_length: int  # actual content length BEFORE batch padding

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
        if t1 >= MAX_DURATION:
            segment_inputs.append(_SegmentInput(
                sdx=sdx, t1=t1, t2=t2, text=text, avg_logprob=avg_logprob,
                eligible=False,
                skip_reason="original start time longer than audio duration",
                fallback=fallback, waveform=None, content_length=0,
            ))
            continue

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)
        waveform_segment = audio[:, f1:f2]
        content_length = waveform_segment.shape[-1]
        # Wav2vec2 needs >=400 samples to produce any frame; pad up if shorter,
        # but pass the true content length so the model masks the padding.
        if content_length < 400:
            waveform_segment = torch.nn.functional.pad(
                waveform_segment, (0, 400 - content_length)
            )

        segment_inputs.append(_SegmentInput(
            sdx=sdx, t1=t1, t2=t2, text=text, avg_logprob=avg_logprob,
            eligible=True, skip_reason=None, fallback=fallback,
            waveform=waveform_segment, content_length=content_length,
        ))

    # 2b. Batched wav2vec2 forward — emissions cached per segment index.
    eligible_indices = [i for i, item in enumerate(segment_inputs) if item.eligible]
    emissions_by_idx: dict[int, np.ndarray] = {}

    batch_size = max(1, int(batch_size))

    def _hf_output_lengths(model: torch.nn.Module, input_lengths: torch.Tensor) -> torch.Tensor:
        """Resolve HuggingFace Wav2Vec2 output frame count per item."""
        for owner in (model, getattr(model, "wav2vec2", None)):
            fn = getattr(owner, "_get_feat_extract_output_lengths", None) if owner is not None else None
            if fn is not None:
                return fn(input_lengths).to(input_lengths.device)
        # Fallback: wav2vec2 base/large stride is 320× downsampling.
        return torch.div(input_lengths, 320, rounding_mode="floor")

    for batch_start in range(0, len(eligible_indices), batch_size):
        batch_idxs = eligible_indices[batch_start:batch_start + batch_size]
        batch = [segment_inputs[i] for i in batch_idxs]

        max_len = max(item.waveform.shape[-1] for item in batch)
        padded = torch.zeros((len(batch), max_len), dtype=batch[0].waveform.dtype)
        for bi, item in enumerate(batch):
            wf = item.waveform[0]
            padded[bi, :wf.shape[-1]] = wf
        padded = padded.to(device)

        content_lengths = torch.as_tensor(
            [item.content_length for item in batch], device=device, dtype=torch.long,
        )

        with torch.inference_mode():
            if model_type == "torchaudio":
                emissions, output_lengths = model(padded, lengths=content_lengths)
            elif model_type == "huggingface":
                attention_mask = torch.zeros(
                    padded.shape[0], padded.shape[-1], device=device, dtype=torch.long,
                )
                for bi, item in enumerate(batch):
                    attention_mask[bi, :item.content_length] = 1
                emissions = model(padded, attention_mask=attention_mask).logits
                output_lengths = _hf_output_lengths(model, content_lengths)
            else:
                raise NotImplementedError(f"Align model of type {model_type} not supported.")
            emissions = torch.log_softmax(emissions, dim=-1)

        emissions_cpu = emissions.cpu().detach()
        # Some torchaudio models / mock objects return None for output_lengths
        # when no length info is available — fall back to the full T dim.
        if output_lengths is None:
            output_lengths_cpu = [emissions_cpu.shape[1]] * emissions_cpu.shape[0]
        else:
            output_lengths_cpu = output_lengths.cpu().tolist()

        for bi, item_idx in enumerate(batch_idxs):
            out_len = max(0, int(output_lengths_cpu[bi]))
            emission = emissions_cpu[bi, :out_len, :]
            emissions_by_idx[item_idx] = np.ascontiguousarray(
                emission.numpy(), dtype=np.float32,
            )

    # 2c. Per-segment Rust DP + post-processing, in original transcript order.
    aligned_segments: List[SingleAlignedSegment] = []
    for item_idx, item in enumerate(segment_inputs):
        sdx = item.sdx

        if not item.eligible:
            logger.warning(
                f'Failed to align segment ("{item.text}"): {item.skip_reason}, '
                'resorting to original'
            )
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
        )

        if rust_subsegments is None:
            logger.warning(
                f'Failed to align segment ("{item.text}"): backtrack failed, '
                'resorting to original'
            )
            aligned_segments.append(item.fallback)
            continue

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

    return {"segments": aligned_segments, "word_segments": word_segments}
