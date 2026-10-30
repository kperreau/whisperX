"""Cached native CPU CTC inference without TorchScript or QTensor creation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from functools import lru_cache
from threading import Lock

import torch
from torch import nn

_cache_load_lock = Lock()
_CACHE_FORMAT_VERSION = 2


@lru_cache(maxsize=1)
def native_int8_supported() -> bool:
    if not torch.backends.mkldnn.is_available():
        return False
    # Seven-bit weights avoid 16-bit pair saturation in AVX2 U8S8 kernels,
    # while retaining the full uint8 activation range.
    weight = torch.tensor([[63] * 256] * 64 + [[-64] * 256] * 64, dtype=torch.int8)
    values = torch.full((16, 256), 255, dtype=torch.uint8)
    try:
        packed = torch.ops.onednn.qlinear_prepack(weight, None)
        output = torch.ops.onednn.qlinear_pointwise.tensor(
            values, torch.tensor(1.), torch.tensor(0, dtype=torch.int64), packed,
            torch.ones(128), torch.zeros(128, dtype=torch.int64), None,
            1.0, 0, torch.float32, "none", [], "",
        )
    except (AttributeError, RuntimeError):
        # A build can include oneDNN without supporting these operators or ISA.
        return False
    return torch.equal(output, values.float() @ weight.float().T)


def materialize_weights(model: nn.Module) -> None:
    """Evaluate inference-only parametrizations once, before saving or loading."""
    for module in model.modules():
        if hasattr(module, "parametrizations"):
            for name in list(module.parametrizations.keys()):
                nn.utils.parametrize.remove_parametrizations(module, name, leave_parametrized=True)


@lru_cache(maxsize=2)
def cache_globals(model_type: str) -> list[object]:
    # Only the pinned model families and ordinary Torch layers are allowed by
    # the weights-only unpickler. Cache metadata cannot request arbitrary imports.
    allowed = [set, Int8Linear, nn.Module, nn.ModuleList, nn.Sequential, nn.Linear,
               nn.Conv1d, nn.GroupNorm, nn.LayerNorm, nn.Dropout, nn.GELU,
               nn.SiLU, nn.ReLU, nn.Identity, nn.ModuleDict,
               nn.functional.gelu, nn.functional.relu, nn.functional.silu]
    if model_type == "torchaudio":
        from torchaudio.models.wav2vec2 import components
        from torchaudio.models.wav2vec2.model import Wav2Vec2Model

        allowed.append(Wav2Vec2Model)
        allowed.extend(value for value in vars(components).values() if
                       isinstance(value, type) and issubclass(value, nn.Module)
                       and value.__module__ == components.__name__)
    elif model_type == "huggingface":
        from transformers import Wav2Vec2Config, activations
        from transformers.models.wav2vec2 import modeling_wav2vec2

        allowed.append(Wav2Vec2Config)
        for module in (modeling_wav2vec2, activations):
            allowed.extend(value for value in vars(module).values() if
                           isinstance(value, type) and issubclass(value, nn.Module)
                           and value.__module__ == module.__name__)
    else:
        raise RuntimeError(f"unsupported alignment model type: {model_type}")
    return allowed


class Int8Linear(nn.Module):
    """Dynamic activation quantization without PyTorch's deprecated QTensor types."""

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(out_features))
        self.register_buffer("bias", torch.empty(out_features) if bias else None)
        self._packed = None

    def prepare(self, *, integer_fallback: bool = False) -> None:
        self._weight_sum = self.qweight.sum(dim=1, dtype=torch.int32)
        if not integer_fallback and native_int8_supported():
            self._packed = torch.ops.onednn.qlinear_prepack(self.qweight, None)
            self._weight_zero = torch.zeros(self.out_features, dtype=torch.int64)
            self._unit_scales = torch.ones(self.out_features)
            self._activation_scale = torch.tensor(1.)
            self._activation_zero = torch.tensor(0, dtype=torch.int64)
        else:
            self._packed = None
            self._weight_transposed = self.qweight.t().contiguous()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shape = (*values.shape[:-1], self.out_features)
        values = values.reshape(-1, self.in_features).contiguous()
        minimum = values.amin(dim=1, keepdim=True).clamp_max(0)
        maximum = values.amax(dim=1, keepdim=True).clamp_min(0)
        scale = ((maximum - minimum) / 255).clamp_min(torch.finfo(torch.float32).eps)
        zero = (-minimum / scale).round().clamp(0, 255)
        quantized = (values / scale).round().add_(zero).clamp_(0, 255).to(torch.uint8)
        if self._packed is not None:
            product = torch.ops.onednn.qlinear_pointwise.tensor(
                quantized, self._activation_scale, self._activation_zero,
                self._packed, self._unit_scales, self._weight_zero, None,
                1.0, 0, torch.float32, "none", [], "",
            )
        else:
            centered = quantized.to(torch.int16).sub_(128).to(torch.int8)
            product = torch._int_mm(centered, self._weight_transposed).float() + 128 * self._weight_sum
        output = (product - zero * self._weight_sum) * (scale * self.weight_scale)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(shape)


class _CachedModel:
    def __init__(self, path: str, cpu_threads: int = 0, *, model_type: str = "torchaudio",
                 integer_fallback: bool | None = None):
        if cpu_threads > 0 and torch.get_num_threads() != cpu_threads:
            torch.set_num_threads(cpu_threads)
        # The explicit allowlist restores model structure without executing its
        # constructors or accepting arbitrary pickle globals. Weights stay mapped.
        with _cache_load_lock, torch.serialization.safe_globals(cache_globals(model_type)):
            cache = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if cache["format"] != _CACHE_FORMAT_VERSION:
            raise RuntimeError("unsupported CPU alignment cache format; rebuild the native bundle")
        if cache["model_type"] != model_type:
            raise RuntimeError("CPU alignment cache model type mismatch")
        self.integer_fallback = not native_int8_supported() if integer_fallback is None else integer_fallback
        self.model = cache["model"]
        self.model.requires_grad_(False).eval()
        if not any(isinstance(layer, Int8Linear) for layer in self.model.modules()):
            raise RuntimeError("alignment cache contains no quantized linear layers")
        self.model_type = model_type
        for module in self.model.modules():
            if isinstance(module, Int8Linear):
                module.prepare(integer_fallback=self.integer_fallback)

    def infer(self, waveforms: torch.Tensor, lengths: torch.Tensor):
        if waveforms.device.type != "cpu":
            raise ValueError("Native int8 alignment supports device='cpu' only")
        waveforms = waveforms.detach().cpu().contiguous()
        lengths = lengths.detach().cpu().contiguous()
        with torch.inference_mode():
            if self.model_type == "torchaudio":
                return self.model(waveforms, lengths=lengths)
            mask = (torch.arange(waveforms.shape[-1]).unsqueeze(0) < lengths.unsqueeze(1)).long()
            emissions = self.model(waveforms, attention_mask=mask).logits
            return emissions, self.model._get_feat_extract_output_lengths(lengths)


class _ReferenceModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, model_type: str):
        super().__init__()
        self.model, self.model_type = model.eval(), model_type

    def forward(self, waveforms, lengths):
        if self.model_type == "torchaudio":
            return self.model(waveforms, lengths=lengths)
        mask = (torch.arange(waveforms.shape[-1]).unsqueeze(0) < lengths.unsqueeze(1)).long()
        return self.model(waveforms, attention_mask=mask).logits, self.model._get_feat_extract_output_lengths(lengths)


def _verification_inputs():
    for batch, samples in ((1, 400), (1, 8000), (2, 16000), (3, 32123)):
        waveforms = torch.randn(batch, samples, generator=torch.Generator().manual_seed(0))
        lengths = torch.tensor([samples // (i + 1) for i in range(batch)])
        yield waveforms, lengths


def _verify_cache(model: torch.nn.Module, path: str, *, integer_fallback: bool | None = None) -> None:
    cached = _CachedModel(path, model_type=getattr(model, "model_type", "torchaudio"),
                          integer_fallback=integer_fallback)
    with torch.inference_mode():
        for waveforms, lengths in _verification_inputs():
            expected, expected_lengths = model(waveforms, lengths)
            actual, actual_lengths = cached.infer(waveforms, lengths)
            torch.testing.assert_close(actual_lengths, expected_lengths)
            if not torch.isfinite(expected).all():
                raise RuntimeError("invalid reference alignment emissions")
            if actual.shape != expected.shape or not torch.isfinite(actual).all():
                raise RuntimeError("invalid quantized alignment emissions")
            for index, length in enumerate(expected_lengths.tolist()):
                divergence = torch.nn.functional.kl_div(
                    actual[index, :length].log_softmax(-1),
                    expected[index, :length].log_softmax(-1),
                    log_target=True, reduction="sum",
                )
                if not torch.isfinite(divergence) or divergence / max(1, length) > 0.1:
                    raise RuntimeError(
                        "quantized CTC posterior divergence exceeds 0.1 nat/frame: "
                        f"{divergence.item() / max(1, length):.4f} for "
                        f"shape={tuple(waveforms.shape)}, item={index}, length={length}"
                    )


def _quantize_linears(model: nn.Module, float_prefixes: tuple[str, ...] = ()) -> int:
    count = 0
    for name, module in list(model.named_modules()):
        module._load_state_dict_pre_hooks.clear()
        module._load_state_dict_post_hooks.clear()
        if not isinstance(module, torch.nn.Linear) or any(
            name == prefix or name.startswith(prefix + ".") for prefix in float_prefixes
        ):
            continue
        weight = module.weight.detach().cpu()
        # Symmetric endpoints avoid giving negative weights a different range.
        scale = (weight.abs().amax(dim=1) / 63).clamp_min(torch.finfo(torch.float32).eps)
        replacement = Int8Linear(module.in_features, module.out_features, module.bias is not None)
        replacement.qweight.copy_((weight / scale[:, None]).round().clamp(-63, 63).to(torch.int8))
        replacement.weight_scale.copy_(scale)
        if module.bias is not None:
            replacement.bias.copy_(module.bias.detach())
        model.set_submodule(name, replacement)
        count += 1
    return count


def _bake_cache(model: torch.nn.Module, model_type: str, path: str, *,
                float_prefixes: tuple[str, ...] = ()) -> None:
    model.eval().cpu().float()
    materialize_weights(model)
    cached = copy.deepcopy(model)
    if not _quantize_linears(cached, float_prefixes):
        raise RuntimeError("alignment cache contains no quantized linear layers")
    torch.save({"format": _CACHE_FORMAT_VERSION, "model_type": model_type,
                "model": cached.requires_grad_(False).eval()}, path)
    reference = _ReferenceModel(model, model_type)
    _verify_cache(reference, path)


class NativeEmissionEngine(_CachedModel):
    """Load a local native int8 bundle without constructing or downloading models."""

    def __init__(self, path, *, threads=0):
        path = Path(path)
        self.metadata = json.loads((path / "metadata.json").read_text())
        if self.metadata.get("format_version") != _CACHE_FORMAT_VERSION or self.metadata.get("type") != "native-int8":
            raise ValueError("Unsupported native int8 alignment bundle format; rebuild the native bundle")
        self.stride = self.metadata["stride"]
        self.receptive_field = self.metadata["receptive_field"]
        super().__init__(str(path / "model.int8.pt"), threads,
                         model_type=self.metadata["model_type"])


def export_native_alignment(model, metadata, output_dir, *, float_prefixes=None):
    """Bake a verified CPU int8 bundle from an unquantized Torch CTC model.

    Known Torchaudio checkpoints keep attention, input and vocabulary projections
    in FP32. English also retains its first encoder block. Feed-forward weights
    use AVX2-safe seven-bit symmetric quantization in int8 storage, with dynamic
    per-token uint8 activations. Both kernels share one profile; export verifies
    the host-selected kernel.
    Callers can explicitly choose float module prefixes for other models.
    Export never downloads a checkpoint and refuses an existing output directory.
    """
    from whisperx.alignment_engine import get_emission_engine

    if metadata["type"] not in {"torchaudio", "huggingface"}:
        raise ValueError("Export requires a Torch torchaudio or huggingface model")
    if any(type(layer).__module__.startswith("torch.ao.nn.quantized") for layer in model.modules()):
        raise ValueError("Export requires an unquantized Torch CTC model")
    if float_prefixes is None:
        # Keep attention and the vocabulary projection in FP32: small score
        # changes can move a CTC boundary even when every word is still present.
        known = {"WAV2VEC2_ASR_BASE_960H", "VOXPOPULI_ASR_BASE_10K_FR"}
        if metadata["type"] == "torchaudio" and metadata.get("model_name") in known:
            prefixes = ["encoder.feature_projection.projection", "aux"]
            prefixes.extend(name for name, _ in model.named_modules() if name.endswith(".attention"))
            if metadata["model_name"] == "WAV2VEC2_ASR_BASE_960H":
                prefixes.append("encoder.transformer.layers.0")
            float_prefixes = tuple(prefixes)
        else:
            float_prefixes = ()
    module_names = dict(model.named_modules())
    if any(prefix not in module_names for prefix in float_prefixes):
        raise ValueError("Float module prefix does not exist in the model")
    geometry = get_emission_engine(model, metadata)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _bake_cache(model, metadata["type"], str(output_dir / "model.int8.pt"),
                float_prefixes=tuple(float_prefixes))
    bundle_metadata = {**metadata, "type": "native-int8", "model_type": metadata["type"],
                       "format_version": _CACHE_FORMAT_VERSION, "stride": geometry.stride,
                       "receptive_field": geometry.receptive_field,
                       "quantization": "native-per-channel-int8-w7-per-token-u8",
                       "weight_qmax": 63, "activation_qmax": 255,
                       "float_prefixes": list(float_prefixes)}
    (output_dir / "metadata.json").write_text(
        json.dumps(bundle_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return output_dir


