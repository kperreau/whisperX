"""CTC emission engines and bounded, overlapping wav2vec2 inference.

Engines return raw logits and exact output lengths. The scheduler owns padding,
log-softmax and stitching, so Torch and ONNX follow the same alignment path.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from whisperx.audio import SAMPLE_RATE


class EmissionEngine(Protocol):
    stride: int
    receptive_field: int

    def infer(self, waveforms: torch.Tensor, lengths: torch.Tensor):
        """Return (batch × frames × vocabulary logits, output lengths)."""
        ...


@dataclass
class TorchEmissionEngine:
    model: object
    model_type: str
    stride: int = 320
    receptive_field: int = 400

    def infer(self, waveforms, lengths):
        if self.model_type == "torchaudio":
            return self.model(waveforms, lengths=lengths)
        if self.model_type != "huggingface":
            raise ValueError(f"Unsupported alignment model type: {self.model_type}")
        mask = (torch.arange(waveforms.shape[-1], device=waveforms.device)[None] < lengths[:, None]).long()
        logits = self.model(waveforms, attention_mask=mask).logits
        output_lengths = None
        for owner in (self.model, getattr(self.model, "wav2vec2", None)):
            fn = getattr(owner, "_get_feat_extract_output_lengths", None)
            if fn is not None:
                output_lengths = fn(lengths)
                break
        if output_lengths is None:
            output_lengths = (lengths - self.receptive_field) // self.stride + 1
        return logits, output_lengths


class OnnxEmissionEngine:
    """Load a local exported bundle without any model-hub access (CPU only)."""

    def __init__(self, path, *, threads=0):
        import onnxruntime as ort

        path = Path(path)
        self.metadata = json.loads((path / "metadata.json").read_text())
        if self.metadata.get("format_version") != 1:
            raise ValueError("Unsupported ONNX alignment bundle format")
        self.stride = self.metadata["stride"]
        self.receptive_field = self.metadata["receptive_field"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path / "model.onnx"), options,
                                           providers=["CPUExecutionProvider"])

    def infer(self, waveforms, lengths):
        if waveforms.device.type != "cpu":
            raise ValueError("ONNX alignment currently supports device='cpu' only")
        logits, output_lengths = self.session.run(
            ["emissions", "output_lengths"],
            {"waveforms": np.ascontiguousarray(waveforms.numpy(), dtype=np.float32),
             "lengths": np.ascontiguousarray(lengths.numpy(), dtype=np.int64)},
        )
        return torch.from_numpy(logits), torch.from_numpy(output_lengths)


def get_emission_engine(model, metadata) -> EmissionEngine:
    if metadata["type"] in {"onnx", "native-int8", "custom"}:
        return model
    stride, receptive_field = 320, 400
    if metadata["type"] == "huggingface":
        config = getattr(model, "config", None)
        kernels = getattr(config, "conv_kernel", None)
        strides = getattr(config, "conv_stride", None)
        if isinstance(kernels, (tuple, list)) and isinstance(strides, (tuple, list)):
            if not kernels or len(kernels) != len(strides):
                raise ValueError("Invalid Hugging Face convolution geometry")
            stride, receptive_field = 1, 1
            for kernel, step in zip(kernels, strides):
                if not isinstance(kernel, Integral) or not isinstance(step, Integral) or min(kernel, step) < 1:
                    raise ValueError("Convolution kernels and strides must be positive integers")
                receptive_field += (kernel - 1) * stride
                stride *= step
            if getattr(config, "add_adapter", False) is True:
                # Padded adapter convolutions preserve the initial frame and
                # apply ceil(length / stride), matching this frame-count grid.
                stride *= config.adapter_stride ** config.num_adapter_layers
    return TorchEmissionEngine(model, metadata["type"], metadata.get("stride", stride),
                               metadata.get("receptive_field", receptive_field))


def infer_emissions(waveforms, engine: EmissionEngine, device, *, batch_size=8,
                    chunk_seconds=10.0, context_seconds=1.0, max_batch_seconds=15.0):
    """Return one log-probability matrix per waveform, in input order.

    Chunk cores and context start on the model's sample stride. Each global
    frame is retained exactly once; overlapping context is discarded. Setting
    chunk_seconds=None disables chunking for reference/fidelity measurements.
    max_batch_seconds limits padded samples per forward, independently of size.
    A chunk can differ from a full-context forward: benchmark word boundaries.
    """
    if (not isinstance(batch_size, Integral) or batch_size < 1 or
            not math.isfinite(max_batch_seconds) or max_batch_seconds <= 0 or
            not math.isfinite(context_seconds) or context_seconds < 0):
        raise ValueError("batch_size/max_batch_seconds must be positive and context_seconds nonnegative")
    stride, receptive = engine.stride, engine.receptive_field
    if not all(isinstance(value, Integral) and value > 0 for value in (stride, receptive)):
        raise ValueError("Engine stride and receptive_field must be positive integers")
    if chunk_seconds is not None and (not math.isfinite(chunk_seconds) or chunk_seconds * SAMPLE_RATE < receptive):
        raise ValueError("chunk_seconds is shorter than the model receptive field")
    core_frames = max(1, int(chunk_seconds * SAMPLE_RATE) // stride) if chunk_seconds else None
    context_frames = int(context_seconds * SAMPLE_RATE) // stride
    waveforms = list(waveforms)
    jobs = []
    whole_waveforms = set()
    parts = [[] for _ in waveforms]
    for index, waveform in enumerate(waveforms):
        waveform = waveform.reshape(-1)
        if not waveform.numel():
            raise ValueError("Cannot infer an empty waveform")
        if waveform.numel() < receptive:
            waveform = torch.nn.functional.pad(waveform, (0, receptive - waveform.numel()))
        frames = (waveform.numel() - receptive) // stride + 1
        # A short clip already fits inside the largest contextual window. Avoid
        # an extra tiny tail forward (and a short-input latency regression).
        step = frames if core_frames is None or frames <= core_frames + 2 * context_frames else core_frames
        if step == frames:
            whole_waveforms.add(index)
        for start in range(0, frames, step):
            end = min(frames, start + step)
            left = max(0, start - context_frames)
            right = min(frames, end + context_frames)
            # Preserve all trailing samples on the last window.
            sample_end = waveform.numel() if right == frames else (right - 1) * stride + receptive
            jobs.append((index, start, start - left, end - left,
                         waveform[left * stride:sample_end]))
    # Similar lengths share a batch, reducing quadratic attention on padding.
    jobs.sort(key=lambda job: job[-1].numel())
    batches = []
    batch = []
    budget = int(max_batch_seconds * SAMPLE_RATE)
    for job in jobs:
        if batch and (len(batch) >= batch_size or job[-1].numel() * (len(batch) + 1) > budget):
            batches.append(batch)
            batch = []
        batch.append(job)
    if batch:
        batches.append(batch)
    with torch.inference_mode():
        for batch in batches:
            padded = torch.nn.utils.rnn.pad_sequence([j[-1] for j in batch], batch_first=True).to(device)
            lengths = torch.tensor([j[-1].numel() for j in batch], device=device)
            logits, output_lengths = engine.infer(padded, lengths)
            if (not torch.is_tensor(logits) or logits.ndim != 3 or
                    logits.shape[0] != len(batch) or min(logits.shape[1:]) < 1):
                raise ValueError("Emission engine must return nonempty batch × frames × vocabulary logits")
            if output_lengths is None:
                output_lengths = [logits.shape[1]] * len(batch)
            else:
                output_lengths = torch.as_tensor(output_lengths)
                if (output_lengths.ndim != 1 or len(output_lengths) != len(batch) or
                        output_lengths.dtype not in {torch.int32, torch.int64}):
                    raise ValueError("Emission output lengths must contain one integer per batch item")
                output_lengths = output_lengths.cpu().tolist()
            probabilities = logits.float().log_softmax(-1).cpu()
            for bi, (index, start, left, right, _) in enumerate(batch):
                available = output_lengths[bi]
                if not 0 < available <= probabilities.shape[1]:
                    raise ValueError("Emission output length must be positive and fit the logits frame dimension")
                if index in whole_waveforms:
                    right = available
                if right > available:
                    raise ValueError(f"Emission engine returned {available} frames; expected at least {right}")
                # Copy: retaining a view would retain the full padded batch.
                emission = probabilities[bi, left:right].numpy().copy()
                if np.isnan(emission).any():
                    raise ValueError("Emission engine returned invalid CTC probabilities (NaN)")
                parts[index].append((start, emission))
    return [np.concatenate([part for _, part in sorted(items)], axis=0) for items in parts]


def export_onnx_alignment(model, metadata, output_dir, *, opset=23, quantize=False):
    """Export a local wav2vec2 bundle; int8 targets constant-weight MatMul only.

    Export the float model, before Torch dynamic quantization. Export requires
    the align-onnx extra. No checkpoint is downloaded by this function.
    """
    import onnx

    if metadata["type"] not in {"torchaudio", "huggingface"}:
        raise ValueError("Export requires a Torch torchaudio or huggingface model")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    engine = get_emission_engine(model, metadata)

    class ExportModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, waveforms, lengths):
            return engine.infer(waveforms, lengths)

    wrapper = ExportModel().eval()
    example = (torch.zeros(2, SAMPLE_RATE), torch.tensor([SAMPLE_RATE, SAMPLE_RATE // 2]))
    path = output_dir / "model.onnx"
    torch.onnx.export(wrapper, example, str(path), input_names=["waveforms", "lengths"],
                      output_names=["emissions", "output_lengths"], dynamo=True,
                      dynamic_shapes=({0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},
                                      {0: torch.export.Dim.DYNAMIC}),
                      opset_version=opset, external_data=False)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        float_path = output_dir / "float.onnx"
        path.rename(float_path)
        quantize_dynamic(str(float_path), str(path), weight_type=QuantType.QInt8,
                         per_channel=True, reduce_range=True, op_types_to_quantize=["MatMul"],
                         extra_options={"MatMulConstBOnly": True})
        float_path.unlink()
    bundle_metadata = {**metadata, "type": "onnx", "format_version": 1,
                       "stride": engine.stride, "receptive_field": engine.receptive_field,
                       "opset": opset, "quantized": quantize,
                       "quantization": "qint8-per-channel-reduced-range" if quantize else "fp32"}
    (output_dir / "metadata.json").write_text(json.dumps(bundle_metadata, indent=2) + "\n")
    return output_dir
