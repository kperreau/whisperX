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
# Version 1 bundles quantized activations per tensor over 8 bits; re-export them.
CACHE_FORMAT_VERSION = 2
# Seven-bit activation codes keep every u8 x s8 pair sum inside the signed 16-bit
# intermediates of AVX2 kernels without VNNI, so oneDNN is exact on every x86 CPU.
ACTIVATION_LEVELS = 127
# Integer matrix products take int8 operands: center the codes around zero.
_CODE_OFFSET = 64

try:
    from whisperx_ext.quant import dequantize_rows as _dequantize_rows
    from whisperx_ext.quant import quantize_rows as _quantize_rows
except ImportError:  # A source checkout without the compiled extension.
    _quantize_rows = _dequantize_rows = None


@lru_cache(maxsize=1)
def native_int8_supported() -> bool:
    if not torch.backends.mkldnn.is_available():
        return False
    # Some ISA kernels reduce signed weights internally or saturate 16-bit pair
    # sums. Check the largest products a bundle can produce once, rather than
    # scanning every model weight at each cold start.
    weight = torch.tensor([[127] * 256] * 64 + [[-128] * 256] * 64, dtype=torch.int8)
    values = torch.full((16, 256), ACTIVATION_LEVELS, dtype=torch.uint8)
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


def quantize_rows_reference(values: torch.Tensor):
    """Torch reference of ``whisperx_ext.quant.quantize_rows``, bit for bit.

    Each row of a float32 matrix gets its own asymmetric 7-bit scale and zero
    point, so one loud frame does not coarsen every other frame's codes.
    """
    minimum = values.amin(dim=1, keepdim=True).clamp_max(0)
    maximum = values.amax(dim=1, keepdim=True).clamp_min(0)
    scale = ((maximum - minimum) / ACTIVATION_LEVELS).clamp_min(torch.finfo(torch.float32).eps)
    scale = torch.where(torch.isfinite(values).all(dim=1, keepdim=True), scale, torch.nan)
    zero = (-minimum / scale).round().clamp(0, ACTIVATION_LEVELS)
    codes = (values / scale).round().add_(zero).clamp_(0, ACTIVATION_LEVELS).to(torch.uint8)
    return codes, scale.view(-1), zero.view(-1)


def dequantize_rows_reference(products, scale, zero, sums, bias):
    """Torch reference of ``whisperx_ext.quant.dequantize_rows``, in place."""
    products.sub_(zero[:, None] * sums).mul_(scale[:, None])
    if bias is not None:
        products.add_(bias)
    return products


def _quantize(values: torch.Tensor):
    if _quantize_rows is None:
        return quantize_rows_reference(values)
    codes = torch.empty(values.shape, dtype=torch.uint8)
    scale, zero = torch.empty(values.shape[0]), torch.empty(values.shape[0])
    _quantize_rows(values.numpy(), codes.numpy(), scale.numpy(), zero.numpy(), torch.get_num_threads())
    return codes, scale, zero


def _dequantize(products, scale, zero, sums, bias):
    if _dequantize_rows is None:
        return dequantize_rows_reference(products, scale, zero, sums, bias)
    _dequantize_rows(products.numpy(), scale.numpy(), zero.numpy(), sums.numpy(),
                     None if bias is None else bias.numpy(), torch.get_num_threads())
    return products


class Int8Linear(nn.Module):
    """Per-row dynamic 7-bit activations times per-channel int8 weights.

    Avoids PyTorch's deprecated QTensor types. oneDNN and the integer fallback
    compute the same exact int32 products; the dequantization is shared.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(out_features))
        self.register_buffer("bias", torch.empty(out_features) if bias else None)
        self._packed = None

    def prepare(self, *, integer_fallback: bool = False) -> None:
        sums = self.qweight.sum(dim=1, dtype=torch.int32)
        # Removes each row's zero point from the products after the GEMM.
        self._weight_sums = sums.float() * self.weight_scale
        if not integer_fallback and native_int8_supported():
            self._packed = torch.ops.onednn.qlinear_prepack(self.qweight, None)
            self._weight_zero = torch.zeros(self.out_features, dtype=torch.int64)
            # Per-row scales and zero points are applied by the dequantization.
            self._unit_scale = torch.tensor(1.)
            self._no_zero = torch.tensor(0, dtype=torch.int64)
        else:
            self._packed = None
            # Development platforms without oneDNN (notably macOS ARM) use
            # the same integer arithmetic, with no legacy quantizer.
            self._weight_transposed = self.qweight.t().contiguous()
            self._code_offset = _CODE_OFFSET * sums

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shape = (*values.shape[:-1], self.out_features)
        values = values.detach().reshape(-1, self.in_features).contiguous()
        codes, scale, zero = _quantize(values)
        if self._packed is not None:
            products = torch.ops.onednn.qlinear_pointwise.tensor(
                codes, self._unit_scale, self._no_zero, self._packed, self.weight_scale,
                self._weight_zero, None, 1.0, 0, torch.float32, "none", [], "",
            )
        else:
            centered = codes.to(torch.int8).sub_(_CODE_OFFSET)
            products = torch._int_mm(centered, self._weight_transposed).add_(self._code_offset)
            products = products.to(torch.float32).mul_(self.weight_scale)
        return _dequantize(products, scale, zero, self._weight_sums, self.bias).reshape(shape)


class _CachedModel:
    def __init__(self, path: str, cpu_threads: int = 0, *, model_type: str = "torchaudio",
                 integer_fallback: bool | None = None, precision: str = "int8"):
        if cpu_threads > 0 and torch.get_num_threads() != cpu_threads:
            torch.set_num_threads(cpu_threads)
        # The explicit allowlist restores model structure without executing its
        # constructors or accepting arbitrary pickle globals. Weights are read
        # into anonymous memory: mapped weights page-fault during the first
        # forward instead, which measured 1.3 s slower to the first emissions
        # (FP32, 10 s of audio) in a gVisor-sandboxed Scaleway job.
        with _cache_load_lock, torch.serialization.safe_globals(cache_globals(model_type)):
            cache = torch.load(path, map_location="cpu", weights_only=True)
        if cache["format"] != CACHE_FORMAT_VERSION:
            raise RuntimeError("unsupported CPU alignment cache format; re-export the native bundle")
        if cache["model_type"] != model_type:
            raise RuntimeError("CPU alignment cache model type mismatch")
        if cache.get("precision", "int8") != precision:
            raise RuntimeError("CPU alignment cache precision mismatch")
        self.model = cache["model"]
        self.model.requires_grad_(False).eval()
        self.model_type = model_type
        self.integer_fallback = None
        if precision == "fp32":
            return
        self.integer_fallback = not native_int8_supported() if integer_fallback is None else integer_fallback
        if not any(isinstance(layer, Int8Linear) for layer in self.model.modules()):
            raise RuntimeError("alignment cache contains no quantized linear layers")
        for module in self.model.modules():
            if isinstance(module, Int8Linear):
                module.prepare(integer_fallback=self.integer_fallback)

    def infer(self, waveforms: torch.Tensor, lengths: torch.Tensor):
        if waveforms.device.type != "cpu":
            raise ValueError("Native alignment supports device='cpu' only")
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


def _verify_cache(model: torch.nn.Module, path: str, *, integer_fallback: bool | None = None,
                  precision: str = "int8") -> None:
    model_type = getattr(model, "model_type", "torchaudio")
    cached = _CachedModel(path, model_type=model_type, integer_fallback=integer_fallback, precision=precision)
    # A host oneDNN kernel must reproduce the exact integer arithmetic.
    exact = (_CachedModel(path, model_type=model_type, integer_fallback=True)
             if cached.integer_fallback is False else None)
    with torch.inference_mode():
        for waveforms, lengths in _verification_inputs():
            expected, expected_lengths = model(waveforms, lengths)
            actual, actual_lengths = cached.infer(waveforms, lengths)
            if exact is not None and not torch.allclose(actual, exact.infer(waveforms, lengths)[0],
                                                        rtol=1e-5, atol=1e-5):
                raise RuntimeError("oneDNN int8 kernel disagrees with exact integer arithmetic "
                                   f"for shape={tuple(waveforms.shape)}")
            if precision == "fp32" and not torch.equal(actual, expected):
                raise RuntimeError("FP32 alignment cache differs from the source model "
                                   f"for shape={tuple(waveforms.shape)}")
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
        if not isinstance(module, torch.nn.Linear) or any(
            name == prefix or name.startswith(prefix + ".") for prefix in float_prefixes
        ):
            continue
        weight = module.weight.detach().cpu()
        # Symmetric endpoints avoid giving negative weights a different range.
        scale = (weight.abs().amax(dim=1) / 127).clamp_min(torch.finfo(torch.float32).eps)
        replacement = Int8Linear(module.in_features, module.out_features, module.bias is not None)
        replacement.qweight.copy_((weight / scale[:, None]).round().clamp(-127, 127).to(torch.int8))
        replacement.weight_scale.copy_(scale)
        if module.bias is not None:
            replacement.bias.copy_(module.bias.detach())
        model.set_submodule(name, replacement)
        count += 1
    return count


def _bake_cache(model: torch.nn.Module, model_type: str, path: str, *,
                float_prefixes: tuple[str, ...] = (), precision: str = "int8") -> None:
    model.eval().cpu().float()
    materialize_weights(model)
    cached = copy.deepcopy(model)
    for module in cached.modules():
        # Checkpoint-compatibility hooks (weight_norm's, for one) are local
        # functions: they cannot be pickled and a bundle never loads a state dict.
        module._load_state_dict_pre_hooks.clear()
        module._load_state_dict_post_hooks.clear()
    if precision == "int8" and not _quantize_linears(cached, float_prefixes):
        raise RuntimeError("alignment cache contains no quantized linear layers")
    torch.save({"format": CACHE_FORMAT_VERSION, "model_type": model_type, "precision": precision,
                "model": cached.requires_grad_(False).eval()}, path)
    _verify_cache(_ReferenceModel(model, model_type), path, precision=precision)


# Bundle type -> (weight precision, model file).
_BUNDLES = {"native-int8": ("int8", "model.int8.pt"), "native-fp32": ("fp32", "model.fp32.pt")}


class NativeEmissionEngine(_CachedModel):
    """Load a local native bundle without constructing or downloading models.

    An int8 bundle runs the int8 linear layers; an FP32 bundle runs the source
    model unchanged, only loading faster than constructing it.
    """

    def __init__(self, path, *, threads=0):
        path = Path(path)
        self.metadata = json.loads((path / "metadata.json").read_text())
        if (self.metadata.get("format_version") != CACHE_FORMAT_VERSION
                or self.metadata.get("type") not in _BUNDLES):
            raise ValueError("Unsupported native alignment bundle format; re-export the bundle")
        self.stride = self.metadata["stride"]
        self.receptive_field = self.metadata["receptive_field"]
        precision, filename = _BUNDLES[self.metadata["type"]]
        super().__init__(str(path / filename), threads,
                         model_type=self.metadata["model_type"], precision=precision)


def export_native_alignment(model, metadata, output_dir, *, float_prefixes=None, precision="int8"):
    """Bake a verified CPU bundle from an unquantized Torch CTC model.

    ``precision="fp32"`` stores the source model itself: loading it maps the
    weights without constructing and randomly initializing a model, and export
    checks that its emissions are bitwise identical to the source model's.

    For int8, weights use symmetric per-channel int8 scales; activations use dynamic
    per-row 7-bit codes, so oneDNN on any x86 CPU (AVX2 without VNNI included)
    and the exact integer fallback compute the same products from one profile.
    Known Torchaudio checkpoints retain sensitive layers in FP32: the first
    encoder block for English base, and the input projection plus a middle encoder
    block for French VoxPopuli. Callers can explicitly choose float module
    prefixes for other models.
    Export never downloads a checkpoint and refuses an existing output directory.
    """
    from whisperx.alignment_engine import get_emission_engine

    if metadata["type"] not in {"torchaudio", "huggingface"}:
        raise ValueError("Export requires a Torch torchaudio or huggingface model")
    if any(type(layer).__module__.startswith("torch.ao.nn.quantized") for layer in model.modules()):
        raise ValueError("Export requires an unquantized Torch CTC model")
    if precision not in {"int8", "fp32"}:
        raise ValueError("Native alignment precision must be int8 or fp32")
    if precision == "fp32":
        if float_prefixes is not None:
            raise ValueError("FP32 bundles keep every layer in FP32; omit float_prefixes")
        float_prefixes = ()
    elif float_prefixes is None:
        profiles = {
            "WAV2VEC2_ASR_BASE_960H": ("encoder.transformer.layers.0",),
            "VOXPOPULI_ASR_BASE_10K_FR": ("encoder.feature_projection.projection",
                                              "encoder.transformer.layers.5"),
        }
        float_prefixes = profiles.get(metadata.get("model_name"), ()) if metadata["type"] == "torchaudio" else ()
    module_names = dict(model.named_modules())
    if any(prefix not in module_names for prefix in float_prefixes):
        raise ValueError("Float module prefix does not exist in the model")
    geometry = get_emission_engine(model, metadata)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    bundle_type = f"native-{precision}"
    _bake_cache(model, metadata["type"], str(output_dir / _BUNDLES[bundle_type][1]),
                float_prefixes=tuple(float_prefixes), precision=precision)
    bundle_metadata = {**metadata, "type": bundle_type, "model_type": metadata["type"],
                       "format_version": CACHE_FORMAT_VERSION, "stride": geometry.stride,
                       "receptive_field": geometry.receptive_field}
    if precision == "int8":
        bundle_metadata.update({"quantization": "native-int8-per-channel-weights-7bit-per-row-activations",
                                "activation_levels": ACTIVATION_LEVELS,
                                "float_prefixes": list(float_prefixes)})
    (output_dir / "metadata.json").write_text(
        json.dumps(bundle_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return output_dir


