"""The Rust fusion preserves the Tensor arithmetic, including FP32 rounding."""
import gc

import numpy as np
import pytest
import torch

quantization = pytest.importorskip("whisperx_ext.quantization")


def inputs(rows=3, columns=7, bias=True):
    generator = np.random.default_rng(42)
    return (
        generator.integers(-2_000_000, 2_000_000, (rows, columns)).astype(np.float32),
        generator.uniform(.0001, .1, (rows, 1)).astype(np.float32),
        generator.integers(0, 256, (rows, 1)).astype(np.float32),
        generator.uniform(.0001, .1, columns).astype(np.float32),
        generator.integers(-100_000, 100_000, columns).astype(np.int32),
        generator.standard_normal(columns).astype(np.float32) if bias else None,
    )


@pytest.mark.parametrize("shape", [(0, 3), (1, 1), (2, 7), (3, 32), (17, 768), (9, 3072)])
@pytest.mark.parametrize("bias", [False, True])
def test_fused_reconstruction_matches_the_separate_fp32_tensor_operations(shape, bias):
    values = inputs(*shape, bias=bias)
    product, scale, zero, channel_scale, sums, offset = (
        torch.from_numpy(value) if value is not None else None for value in values
    )
    expected = (product - zero * sums) * (scale * channel_scale)
    if offset is not None:
        expected = expected + offset
    actual = quantization.dequantize_rows(*values)
    assert actual.dtype == np.float32 and actual.shape == shape
    np.testing.assert_array_equal(actual.view(np.uint32), expected.numpy().view(np.uint32))
    assert not np.shares_memory(actual, values[0])


def test_output_owns_its_storage_after_inputs_are_released():
    values = inputs()
    output = torch.from_numpy(quantization.dequantize_rows(*values))
    expected = output.clone()
    del values
    gc.collect()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("index,replacement", [
    (0, np.ones((3, 7), dtype=np.float64)),
    (1, np.ones((2, 1), dtype=np.float32)),
    (2, np.ones((3, 2), dtype=np.float32)),
    (3, np.ones(6, dtype=np.float32)),
    (4, np.ones(8, dtype=np.int32)),
    (5, np.ones(6, dtype=np.float32)),
])
def test_invalid_dimensions_or_dtypes_fail_before_computation(index, replacement):
    values = list(inputs())
    values[index] = replacement
    with pytest.raises((TypeError, ValueError)):
        quantization.dequantize_rows(*values)


@pytest.mark.parametrize("index", range(6))
def test_noncontiguous_inputs_are_rejected(index):
    values = list(inputs())
    value = values[index]
    storage = np.zeros((*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype)
    strided = storage[..., ::2]
    strided[...] = value
    values[index] = strided
    with pytest.raises(ValueError, match="contiguous"):
        quantization.dequantize_rows(*values)


def test_readonly_inputs_are_supported_without_modification():
    values = inputs()
    before = [value.copy() if value is not None else None for value in values]
    for value in values:
        if value is not None:
            value.flags.writeable = False
    quantization.dequantize_rows(*values)
    for value, expected in zip(values, before):
        if value is not None:
            np.testing.assert_array_equal(value, expected)
