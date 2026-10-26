# Alignment engines, memory and timestamp fidelity

`align()` now schedules CTC inference through `EmissionEngine.infer(waveforms,
lengths)`, independently of the Rust trellis and timestamp reconstruction.
Torch Torchaudio, Torch Hugging Face and local ONNX bundles use the same padding,
log-softmax, frame stitching and native alignment path. Existing model/metadata
pairs still work.

## Bounded inference

The default is a 10-second core with one second of left/right context, aligned
to the wav2vec2 convolution stride (320 samples, 400-sample receptive field).
The scheduler discards context emissions and retains every original frame once.
Clips fitting in a contextual window use one forward. Short inputs are padded to
one receptive field; their timestamps still use the real audio duration.

Length sorting and a 15-second padded-sample budget limit batch waste; a batch
also respects `batch_size`. One individual window can exceed the batch budget.
Set `chunk_seconds=None` to measure full context. Hugging Face convolution
geometry and adapter downsampling are derived from the model config. For other
custom CTC geometry, metadata can specify `stride` and `receptive_field`; custom engines expose both.
The implementation targets the standard wav2vec2 CTC convolution stack.

Chunking changes the transformer's context and can move word boundaries. It is a
memory/latency tradeoff, not bitwise equivalence to full-context inference. The
benchmark checks the actual final word sequence and every start/end boundary,
with a configurable 100 ms maximum relative drift budget. This measures agreement
with a reference engine, not error against human-annotated ground truth.

## Export and offline loading

Install `align-onnx` in addition to the normal build dependencies. Export the
float Torch model, before applying Torch dynamic quantization:

```python
import whisperx
from whisperx.alignment_engine import export_onnx_alignment

model, metadata = whisperx.load_align_model("en", "cpu", quantize=False)
export_onnx_alignment(model, metadata, "/tmp/en-float")
export_onnx_alignment(model, metadata, "/tmp/en-int8", quantize=True)

model, metadata = whisperx.load_align_model(
    "en", "cpu", engine="onnx", model_name="/tmp/en-int8", onnx_threads=4,
)
result = whisperx.align(segments, model, metadata, audio, "cpu")
```

Export creates a new directory with `model.onnx` and `metadata.json`. It uses the
[Dynamo ONNX exporter](https://docs.pytorch.org/docs/stable/onnx) with opset 23,
dynamic batch/sample dimensions and runtime graph optimizations. The measured
base models contain 12 ONNX `Attention` operators. Targeted int8 quantization is
per-channel QInt8 with reduced range for constant-weight MatMul nodes; convolution and attention
activation products remain float. See the
[ONNX Runtime quantization documentation](https://onnxruntime.ai/docs/how-to/quantization.html).
Export itself does not fetch checkpoints. Loading a bundle is local and CPU only;
its language must match the request. Quantization is selected at export time.
The CLI's `--threads` limit is passed to ONNX as well as Torch. Invalid engine
shapes, lengths and NaN probabilities raise before entering native alignment;
the native entry point also validates token IDs, display indices and spans.

`model_cache_only=True` now checks the Torchaudio checkpoint before calling its
loader, as well as using `local_files_only` on the Hugging Face path. Missing
local files raise instead of triggering a download. Torch models are set to eval
mode. Torch int8 engine selection favors x86/fbgemm on x86 and qnnpack on ARM.
The CLI defaults to float: measured dynamic int8 is slower on this ARM CPU and
exceeds the French timestamp drift budget. Explicit `--align_quantize True`
remains available after validation on the target CPU.

## Cached native CPU inference

`native-int8` bundles store ordinary integer/float weights and model structure.
They use native Torch convolution/attention and oneDNN linear kernels, without
TorchScript or deprecated PyTorch QTensor creation. Loading uses memory mapping
and a weights-only unpickler with an explicit allowlist of model classes. It
does not construct a model, quantize weights or contact a model hub at startup.

```python
from whisperx.alignment_native import export_native_alignment

model, metadata = whisperx.load_align_model("en", "cpu", quantize=False)
export_native_alignment(model, metadata, "/tmp/en-native")
engine, metadata = whisperx.load_align_model(
    "en", "cpu", engine="native-int8", model_name="/tmp/en-native", cpu_threads=2,
)
result = whisperx.align(segments, engine, metadata, audio, "cpu", chunk_seconds=None)
```

Export refuses an existing directory and writes the manifest after verification.
Each valid batch item must preserve frame counts, produce finite emissions and
remain within 0.1 nat/frame of the source FP32 CTC posterior on fixed short,
variable-duration and unequal-length inputs. This numerical check does not replace
real-speech timestamp and deployment-memory tests.

Weights use symmetric per-channel int8 scales and activations use dynamic uint8
scales. The English Torchaudio `WAV2VEC2_ASR_BASE_960H` profile keeps its first
encoder block in FP32 to preserve word boundaries. The French Torchaudio
`VOXPOPULI_ASR_BASE_10K_FR` profile keeps its input projection in FP32, including
for long narrations. Other models can select float module prefixes through `export_native_alignment(..., float_prefixes=(...))`.
A one-time arithmetic probe checks full-range oneDNN products on the host CPU;
hosts without a correct kernel use exact integer matrix multiplication instead.
Speed depends on the CPU and kernel. Benchmark the deployment hardware.

Rebuild bundles when changing the pinned Torch, TorchAudio or Transformers stack.
CPU cache format version 1 is independent of ONNX bundles. The CLI accepts
`--align_engine native-int8 --align_model /tmp/en-native`; pass
`--align_chunk_seconds 0` for full context. The benchmark helper accepts
`--native-int8 /tmp/en-native` alongside its existing Torch/ONNX variants.

## Spoken text, displayed text and diagnostics

The application supplies the actual pronunciation; WhisperX does not infer how
prices, surfaces, abbreviations or numbers should be read. An explicit span maps
its full spoken duration back onto the displayed word:

```python
segments = [{
    "start": 0.0, "end": 4.0, "text": "cost 43 euros",
    "spoken_replacements": [{"start": 5, "end": 7, "text": "forty three"}],
}]
result = whisperx.align(
    segments, model, metadata, audio, "cpu",
    return_diagnostics=True, unsupported_policy="error",
)
```

Offsets are Python Unicode character offsets, with an exclusive end. Spans must
be nonempty, nonoverlapping and fit within one displayed word. For a price with
spaces, provide a pronunciation for each displayed token. The caller must send
the expanded text to TTS too. The Rust kernel aggregates repeated display indices
so `43` spans **both** `forty` and `three`. Its per-character output assigns the
whole interval to the first display character; the other digits have no independent
phonetic boundaries. Do not interpret these as individual digit timestamps.

`return_diagnostics=True` adds per-input-segment diagnostics: unsupported character
positions, word methods (`ctc`, `spoken`, `wildcard`, `interpolated`, `unaligned`),
word scores, context limitation, frame counts and fallback reasons. Scores are CTC
scores, not calibrated probabilities of correct boundaries. Default wildcard
behavior is preserved; `unsupported_policy="error"` rejects unsupported characters
before inference. Punctuation absent from the dictionary is also unsupported.
CLI equivalents are `--align_diagnostics` and `--align_strict`; spoken span mapping
is currently a Python API feature.

## Reproduce the CPU benchmark

The harness uses a fresh worker for each engine/duration, one warmup and three
measured iterations, four intra-op threads, and the same checkpoint/audio/text.
It reports separate load, model forward, Rust and remaining pipeline timings,
process peak RSS including loading, all final word timestamps and their drift.
Failed/killed/timed-out workers are recorded. `--check` returns nonzero on a worker
failure or a timestamp/memory budget violation; `--max-rss-mib` enables an optional
process RSS budget. Run on Linux or macOS. Run performance cases sequentially.

Download the [pinned JFK fixture](https://raw.githubusercontent.com/openai/whisper/31243bad24cc746f07d4c8bfdd2d974872cb1803/tests/jfk.flac)
used by `tools/check_inference.py`, then:

```sh
python tools/benchmark_alignment.py \
  --audio /tmp/jfk.flac \
  --text 'and so my fellow americans ask not what your country can do for you ask what you can do for your country' \
  --onnx-fp32 /tmp/en-float --onnx-int8 /tmp/en-int8 \
  --durations 0,30,90 --threads 4 --repeats 3 --warmup 1 \
  --output /tmp/alignment.json --check
```

`0` keeps the original clip. Long cases repeat complete utterances and pad the
remaining duration with silence, rather than truncating speech while keeping its
text. The French case uses Homiclip's existing sample narration and its explicitly
normalized spoken transcript, on the French VoxPopuli model. Both fixture hashes
and transcript hashes are in the committed evidence. These small fixtures are
regression probes, not a representative production corpus.

Committed JSON summaries retain timing samples, environment, hashes, word counts,
timestamp hashes, fidelity metrics and graph/bundle information. Rerunning the
harness writes the full timestamps and per-word diagnostics. The profiling helper
also supports both engines; its inclusive total is counted once, and it no longer
patches the model class globally. Its temporary Rust timer should be used in an
isolated process.

## Measurements (2026-10-26)

The tables and evidence below describe native macOS ARM execution with Python
3.15.1, PyTorch 2.14.1 and ONNX Runtime 1.31.0. Model forward dominates; the Rust
kernel remains a few milliseconds on the 90-second cases. ONNX is optional:
opset 23 and int8 do not automatically outperform native Torch on this CPU.

| Language | Engine | Context | Align ms | Forward ms | Peak RSS MiB | Max drift ms | ≤100 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| en | torch-fp32 | full | 2362 | 2357 | 3979 | 0 | pass |
| en | torch-int8 | full | 4024 | 4019 | 4163 | 20 | pass |
| en | torch-fp32 | 10 s cores | 1533 | 1527 | 1506 | 20 | pass |
| en | onnx-fp32 | full | 6119 | 6114 | 4995 | 0 | pass |
| en | onnx-fp32 | 10 s cores | 4798 | 4790 | 1065 | 20 | pass |
| en | onnx-int8 | full | 4700 | 4695 | 4704 | 0 | pass |
| en | onnx-int8 | 10 s cores | 2996 | 2990 | 816 | 20 | pass |
| fr | torch-fp32 | full | 2345 | 2338 | 3979 | 0 | pass |
| fr | torch-int8 | full | 4025 | 4018 | 4164 | 240 | FAIL |
| fr | torch-fp32 | 10 s cores | 1552 | 1543 | 1509 | 80 | pass |
| fr | onnx-fp32 | full | 6288 | 6280 | 5020 | 0 | pass |
| fr | onnx-fp32 | 10 s cores | 4745 | 4736 | 1089 | 80 | pass |
| fr | onnx-int8 | full | 4770 | 4763 | 4725 | 40 | pass |
| fr | onnx-int8 | 10 s cores | 2996 | 2988 | 839 | 80 | pass |

For en, Torch fp32 chunking is **1.54× faster** and reduces peak RSS by **62%**, with maximum relative boundary drift of **20 ms**.

For fr, Torch fp32 chunking is **1.51× faster** and reduces peak RSS by **62%**, with maximum relative boundary drift of **80 ms**.

On shorter clips, savings can be smaller or latency can increase: the French 23.88-second probe is slower with chunking. Torch int8 reaches **240 ms** maximum drift on the 90-second French probe. Reduced-range ONNX int8 reaches **240 ms** maximum drift across the French cases. Quantized variants require corpus validation; inspect each row against the 100 ms budget.

Evidence: [English](benchmarks/alignment-en.json), [French](benchmarks/alignment-fr.json), [2 GiB container check](benchmarks/alignment-2g.json). The benchmark reports model forward separately from Rust/overhead and model loading. Process peak RSS includes loading; it is not just incremental tensor memory.

## Enforced 2 GiB memory check

The same 90-second English ONNX int8 input was run in a local Linux amd64 image,
emulated on macOS ARM, with `--memory 2g --memory-swap 2g --cpus 4 --network none`.
The container received this checkout's Python package and benchmark script through
read-only mounts; its existing native extension handled the unchanged CTC path
(no spoken replacements in this memory case). The changed native span aggregation
is built and covered separately by the local tests.

The chunked run exits 0, records a cgroup limit of 2,147,483,648 bytes and no OOM
kills, and its final word boundaries are checked against the native fp32 reference. Full-context inference with the identical bundle exits 137 and Docker
reports `OOMKilled=true`. See `benchmarks/alignment-2g.json` for the exact image ID,
limits, container states and successful worker metrics. Emulated timings are not
used in the performance tables. A native Scaleway amd64 benchmark and downstream
Homiclip deployment validation remain integration work; this PR publishes no new
release or deployment.

To repeat using an image with Python, Torch, ONNX Runtime, FFmpeg and the native
extension, use the committed helper. It mounts this checkout read-only, starts
both containers sequentially, compares final word boundaries with the native
benchmark reference, saves worker outputs and inspected resource
limits/OOM state, then removes the containers:

```sh
python tools/check_alignment_memory.py \
  --image homiclip-transcribe-onnx:local \
  --audio /tmp/jfk.flac --bundle /tmp/en-int8 \
  --text 'and so my fellow americans ask not what your country can do for you ask what you can do for your country' \
  --reference-json /tmp/alignment.json --output-dir /tmp/alignment-memory
```

The image tag is local; the evidence pins its image ID. Override `--image`,
`--python` or `--platform` for another compatible runtime. The helper returns
nonzero if chunked inference fails or exceeds the timestamp budget;
full-context success or OOM is preserved as
observed evidence, since a different model/runtime can fit the memory limit.

The cross-platform check caught a full-range QInt8 regression: memory stayed below
2 GiB but final timestamps were unusable on this x86 runtime. See the retained
[failed fidelity evidence](benchmarks/alignment-2g-full-range-regression.json).
Reduced-range quantization corrects this case. This is consistent with the
[documented U8S8 saturation risk on AVX2/AVX512](https://onnxruntime.ai/docs/how-to/quantization.html#data-type-selection)
and is why export now uses reduced range by default. Native ARM timings alone
would not have detected this target-runtime failure.
