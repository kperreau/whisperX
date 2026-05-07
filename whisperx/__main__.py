import argparse
import importlib.metadata
import platform

import torch

from whisperx.utils import (LANGUAGES, TO_LANGUAGE_CODE, optional_float,
                            optional_int, str2bool)
from whisperx.log_utils import setup_logging


def cli():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="small", help="name of the Whisper model to use")
    parser.add_argument("--model_cache_only", type=str2bool, default=False, help="If True, will not attempt to download models, instead using cached models from --model_dir")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device type to use for PyTorch inference (e.g. cpu, cuda)")
    parser.add_argument("--device_index", default=0, type=int, help="device index to use for FasterWhisper inference")
    parser.add_argument("--batch_size", default=8, type=int, help="the preferred batch size for inference")
    parser.add_argument("--compute_type", default="default", type=str, choices=["default", "float16", "float32", "int8"], help="compute type for computation; 'default' uses float16 on GPU, int8 on CPU (2-3x faster than float32, ~0.5pt WER cost)")

    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"], help="format of the output file; if not specified, all available formats will be produced")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")
    parser.add_argument("--log-level", type=str, default=None, choices=["debug", "info", "warning", "error", "critical"], help="logging level (overrides --verbose if set)")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    # alignment params
    parser.add_argument("--align_model", default=None, help="Name of phoneme-level ASR model to do alignment")
    parser.add_argument("--align_batch_size", type=int, default=None, help="Batch size for the wav2vec2 alignment forward pass. Higher = faster on GPU, more VRAM. Set to 1 to disable batching. Default: 16 on CPU, 8 on GPU.")
    parser.add_argument("--align_quantize", type=str2bool, default=None, help="Apply dynamic int8 quantization to the wav2vec2 alignment model (CPU only). Typically 2-3x faster on CPU at a tiny accuracy cost. Default: True on CPU, False (ignored) on CUDA.")
    parser.add_argument("--interpolate_method", default="nearest", choices=["nearest", "linear", "ignore"], help="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring.")
    parser.add_argument("--no_align", action='store_true', help="Do not perform phoneme alignment")
    parser.add_argument("--return_char_alignments", action='store_true', help="Return character-level alignments in the output json file")

    # vad params
    parser.add_argument("--vad_method", type=str, default=None, choices=["pyannote", "silero"], help="VAD method to be used. Default: 'silero' on CPU (3-5x faster), 'pyannote' on GPU.")
    parser.add_argument("--vad_onset", type=float, default=0.500, help="Onset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected")
    parser.add_argument("--vad_offset", type=float, default=0.363, help="Offset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected.")
    parser.add_argument("--chunk_size", type=int, default=30, help="Chunk size for merging VAD segments. Default is 30, reduce this if the chunk is too long.")

    # diarization params
    parser.add_argument("--diarize", action="store_true", help="Apply diarization to assign speaker labels to each segment/word")
    parser.add_argument("--min_speakers", default=None, type=int, help="Minimum number of speakers to in audio file")
    parser.add_argument("--max_speakers", default=None, type=int, help="Maximum number of speakers to in audio file")
    parser.add_argument("--diarize_model", default="pyannote/speaker-diarization-community-1", type=str, help="Name of the speaker diarization model to use")
    parser.add_argument("--speaker_embeddings", action="store_true", help="Include speaker embeddings in JSON output (only works with --diarize)")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=None, help="number of beams in beam search, only applicable when temperature is zero. Default: 1 on CPU (greedy, ~1.7x faster), 5 on GPU.")
    parser.add_argument("--patience", type=float, default=1.0, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--suppress_numerals", action="store_true", help="whether to suppress numeric symbols and currency symbols during sampling, since wav2vec2 cannot align them correctly")

    parser.add_argument("--initial_prompt", type=str, default=None, help="soft prompt placed before the audio (first window only). May skip the start of the audio if the prompt overlaps it. Use --prefix for true force-decoding.")
    parser.add_argument("--prefix", type=str, default=None, help="force-decoded prefix: tokens the model is forced to emit at the start of the FIRST window. Use to make the transcript begin with an exact phrase. Truncated to ~224 tokens by faster-whisper. Disables hotwords on that window. Recommended combo for long audio: --prefix TEXT --auto_hotwords TEXT.")
    parser.add_argument("--hotwords", type=str, default=None, help="comma-separated bias terms (vocabulary biasing). Mild but applied to ALL windows and never causes skips. Ignored on windows where --prefix applies.")
    parser.add_argument("--auto_hotwords", type=str, default=None, help="free-form text from which hotwords are extracted automatically and merged into --hotwords. Applies to all windows. Pair with --prefix for force-decoding the first chunk + biasing the rest.")
    parser.add_argument("--auto_hotwords_max", type=int, default=30, help="cap on the number of hotwords extracted by --auto_hotwords (default 30)")
    parser.add_argument("--auto_hotwords_mode", type=str, default="formatted", choices=["names", "formatted", "all"], help="extraction strategy for --auto_hotwords. 'names' = proper nouns + units + acronyms only; 'formatted' (default) = above + digit-bearing tokens and '<digits> <digits>' bigrams so spaced numbers like '1 590' survive; 'all' = above + every word (closest to --initial_prompt biasing strength but accepts a small first-sentence-skip risk).")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=False, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")

    parser.add_argument("--max_line_width", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of lines in a segment")
    parser.add_argument("--highlight_words", type=str2bool, default=False, help="(not possible with --no_align) underline each word as it is spoken in srt and vtt")
    parser.add_argument("--segment_resolution", type=str, default="sentence", choices=["sentence", "chunk"], help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")

    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Access Token to access PyAnnote gated models")

    parser.add_argument("--print_progress", type=str2bool, default = False, help = "if True, progress will be printed in transcribe() and align() methods.")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {importlib.metadata.version('whisperx')}",help="Show whisperx version information and exit")
    parser.add_argument("--python-version", "-P", action="version", version=f"Python {platform.python_version()} ({platform.python_implementation()})",help="Show python version information and exit")
    # fmt: on

    args = parser.parse_args().__dict__

    log_level = args.get("log_level")
    verbose = args.get("verbose")

    if log_level is not None:
        setup_logging(level=log_level)
    elif verbose:
        setup_logging(level="info")
    else:
        setup_logging(level="warning")

    from whisperx.transcribe import transcribe_task

    transcribe_task(args, parser)


if __name__ == "__main__":
    cli()
