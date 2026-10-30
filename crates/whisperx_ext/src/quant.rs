//! Fused per-row activation quantization for native int8 linear layers.
//!
//! Seven-bit codes (0..=127) keep every u8 × s8 pair sum inside the signed
//! 16-bit intermediates of AVX2 kernels without VNNI, so oneDNN computes the
//! same exact integer products on every x86 CPU. Rounding and operation order
//! match the Torch reference in `whisperx.alignment_native` bit for bit; each
//! function replaces several full passes of Torch element-wise operators.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const LEVELS: f32 = 127.0;
const LANES: usize = 8;

/// Quantize one row and return its (scale, zero point). Non-finite inputs
/// produce a NaN scale, which the dequantization turns into NaN outputs.
#[inline(always)]
fn quantize_row(row: &[f32], codes: &mut [u8]) -> (f32, f32) {
    // Lane-wise accumulators vectorize without reassociating float math.
    let mut low = [0.0f32; LANES];
    let mut high = [0.0f32; LANES];
    let mut poison = [0.0f32; LANES];
    let (chunks, remainder) = row.as_chunks::<LANES>();
    for chunk in chunks {
        for lane in 0..LANES {
            let value = chunk[lane];
            low[lane] = if value < low[lane] { value } else { low[lane] };
            high[lane] = if value > high[lane] { value } else { high[lane] };
            poison[lane] += value * 0.0;
        }
    }
    let (mut minimum, mut maximum, mut invalid) = (0.0f32, 0.0f32, 0.0f32);
    for lane in 0..LANES {
        minimum = if low[lane] < minimum { low[lane] } else { minimum };
        maximum = if high[lane] > maximum { high[lane] } else { maximum };
        invalid += poison[lane];
    }
    for &value in remainder {
        minimum = if value < minimum { value } else { minimum };
        maximum = if value > maximum { value } else { maximum };
        invalid += value * 0.0;
    }
    let scale = if invalid == 0.0 {
        ((maximum - minimum) / LEVELS).max(f32::EPSILON)
    } else {
        f32::NAN
    };
    let zero = (-minimum / scale).round_ties_even().clamp(0.0, LEVELS);
    for (code, &value) in codes.iter_mut().zip(row) {
        *code = ((value / scale).round_ties_even() + zero).clamp(0.0, LEVELS) as u8;
    }
    (scale, zero)
}

#[inline(always)]
fn dequantize_row(row: &mut [f32], scale: f32, zero: f32, sums: &[f32], bias: Option<&[f32]>) {
    match bias {
        Some(bias) => {
            for ((value, &sum), &offset) in row.iter_mut().zip(sums).zip(bias) {
                *value = (*value - zero * sum) * scale + offset;
            }
        }
        None => {
            for (value, &sum) in row.iter_mut().zip(sums) {
                *value = (*value - zero * sum) * scale;
            }
        }
    }
}

#[inline(always)]
fn quantize_block(values: &[f32], codes: &mut [u8], scales: &mut [f32], zeros: &mut [f32], width: usize) {
    for (((row, codes), scale), zero) in values
        .chunks_exact(width)
        .zip(codes.chunks_exact_mut(width))
        .zip(scales.iter_mut())
        .zip(zeros.iter_mut())
    {
        (*scale, *zero) = quantize_row(row, codes);
    }
}

#[inline(always)]
fn dequantize_block(values: &mut [f32], scales: &[f32], zeros: &[f32], sums: &[f32], bias: Option<&[f32]>) {
    for ((row, &scale), &zero) in values.chunks_exact_mut(sums.len()).zip(scales).zip(zeros) {
        dequantize_row(row, scale, zero, sums, bias);
    }
}

// The generic bodies are inlined into these AVX2 clones, which the runtime
// dispatch below selects; other targets use their baseline vector ISA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_block_avx2(values: &[f32], codes: &mut [u8], scales: &mut [f32], zeros: &mut [f32], width: usize) {
    quantize_block(values, codes, scales, zeros, width)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_block_avx2(values: &mut [f32], scales: &[f32], zeros: &[f32], sums: &[f32], bias: Option<&[f32]>) {
    dequantize_block(values, scales, zeros, sums, bias)
}

fn quantize_dispatch(values: &[f32], codes: &mut [u8], scales: &mut [f32], zeros: &mut [f32], width: usize) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU supports AVX2, checked just above.
        return unsafe { quantize_block_avx2(values, codes, scales, zeros, width) };
    }
    quantize_block(values, codes, scales, zeros, width)
}

fn dequantize_dispatch(values: &mut [f32], scales: &[f32], zeros: &[f32], sums: &[f32], bias: Option<&[f32]>) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU supports AVX2, checked just above.
        return unsafe { dequantize_block_avx2(values, scales, zeros, sums, bias) };
    }
    dequantize_block(values, scales, zeros, sums, bias)
}

/// Rows per worker; smaller inputs stay on the calling thread.
fn rows_per_worker(rows: usize, threads: usize) -> usize {
    const MIN_ROWS: usize = 64;
    rows.div_ceil(threads.max(1)).max(MIN_ROWS)
}

fn to_value_error<E: std::fmt::Display>(error: E) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// `as_slice` also accepts Fortran order, which would silently transpose rows.
fn require_c_order(arrays: &[bool]) -> PyResult<()> {
    if arrays.iter().all(|&c_order| c_order) {
        Ok(())
    } else {
        Err(PyValueError::new_err("arrays must be C-contiguous"))
    }
}

/// Quantize each row of a C-contiguous float32 matrix to 7-bit codes.
///
/// Writes `codes[i, j] = clamp(round(values[i, j] / scale[i]) + zero[i], 0, 127)`
/// with `scale[i] = max((max(row, 0) - min(row, 0)) / 127, eps)` and
/// `zero[i] = clamp(round(-min(row, 0) / scale[i]), 0, 127)`, rounding half to even.
#[pyfunction]
#[pyo3(signature = (values, codes, scales, zeros, threads = 1))]
fn quantize_rows<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f32>,
    mut codes: PyReadwriteArray2<'py, u8>,
    mut scales: PyReadwriteArray1<'py, f32>,
    mut zeros: PyReadwriteArray1<'py, f32>,
    threads: usize,
) -> PyResult<()> {
    let (rows, width) = values.as_array().dim();
    if codes.as_array().dim() != (rows, width) || scales.len() != rows || zeros.len() != rows {
        return Err(PyValueError::new_err("codes, scales and zeros must match the value rows"));
    }
    if width == 0 {
        return Err(PyValueError::new_err("values must have at least one column"));
    }
    require_c_order(&[values.is_c_contiguous(), codes.is_c_contiguous(), scales.is_c_contiguous(),
                      zeros.is_c_contiguous()])?;
    let values = values.as_slice().map_err(to_value_error)?;
    let codes = codes.as_slice_mut().map_err(to_value_error)?;
    let scales = scales.as_slice_mut().map_err(to_value_error)?;
    let zeros = zeros.as_slice_mut().map_err(to_value_error)?;
    py.detach(|| {
        let step = rows_per_worker(rows, threads);
        if step >= rows {
            return quantize_dispatch(values, codes, scales, zeros, width);
        }
        std::thread::scope(|scope| {
            for (((values, codes), scales), zeros) in values
                .chunks(step * width)
                .zip(codes.chunks_mut(step * width))
                .zip(scales.chunks_mut(step))
                .zip(zeros.chunks_mut(step))
            {
                scope.spawn(move || quantize_dispatch(values, codes, scales, zeros, width));
            }
        });
    });
    Ok(())
}

/// Turn integer products into outputs in place.
///
/// `values[i, j]` holds `sum_k codes[i, k] * qweight[j, k] * weight_scale[j]`;
/// it becomes `(values[i, j] - zeros[i] * sums[j]) * scales[i] + bias[j]`, with
/// `sums[j] = sum_k qweight[j, k] * weight_scale[j]`.
#[pyfunction]
#[pyo3(signature = (values, scales, zeros, sums, bias = None, threads = 1))]
fn dequantize_rows<'py>(
    py: Python<'py>,
    mut values: PyReadwriteArray2<'py, f32>,
    scales: PyReadonlyArray1<'py, f32>,
    zeros: PyReadonlyArray1<'py, f32>,
    sums: PyReadonlyArray1<'py, f32>,
    bias: Option<PyReadonlyArray1<'py, f32>>,
    threads: usize,
) -> PyResult<()> {
    let (rows, width) = values.as_array().dim();
    if scales.len() != rows || zeros.len() != rows || sums.len() != width
        || bias.as_ref().is_some_and(|bias| bias.len() != width)
    {
        return Err(PyValueError::new_err("scales/zeros must match rows and sums/bias must match columns"));
    }
    if width == 0 {
        return Err(PyValueError::new_err("values must have at least one column"));
    }
    require_c_order(&[values.is_c_contiguous(), scales.is_c_contiguous(), zeros.is_c_contiguous(),
                      sums.is_c_contiguous(), bias.as_ref().is_none_or(|bias| bias.is_c_contiguous())])?;
    let values = values.as_slice_mut().map_err(to_value_error)?;
    let scales = scales.as_slice().map_err(to_value_error)?;
    let zeros = zeros.as_slice().map_err(to_value_error)?;
    let sums = sums.as_slice().map_err(to_value_error)?;
    let bias = match bias.as_ref() {
        Some(bias) => Some(bias.as_slice().map_err(to_value_error)?),
        None => None,
    };
    py.detach(|| {
        let step = rows_per_worker(rows, threads);
        if step >= rows {
            return dequantize_dispatch(values, scales, zeros, sums, bias);
        }
        std::thread::scope(|scope| {
            for ((values, scales), zeros) in values
                .chunks_mut(step * width)
                .zip(scales.chunks(step))
                .zip(zeros.chunks(step))
            {
                scope.spawn(move || dequantize_dispatch(values, scales, zeros, sums, bias));
            }
        });
    });
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quantize_rows, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_rows, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_includes_zero_and_rounds_half_to_even() {
        let row = [0.5f32, 1.0, 2.0, 127.0];
        let mut codes = [0u8; 4];
        let (scale, zero) = quantize_row(&row, &mut codes);
        assert_eq!((scale, zero), (1.0, 0.0));
        assert_eq!(codes, [0, 1, 2, 127]);
    }

    #[test]
    fn non_finite_rows_poison_the_scale() {
        let mut codes = [0u8; 9];
        let (scale, _) = quantize_row(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, f32::INFINITY, 1.0], &mut codes);
        assert!(scale.is_nan());
        let mut row = [1.0f32, 2.0];
        dequantize_row(&mut row, scale, 0.0, &[1.0, 1.0], None);
        assert!(row.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn negative_rows_use_the_full_range() {
        let row = [-127.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -63.4];
        let mut codes = [0u8; 9];
        let (scale, zero) = quantize_row(&row, &mut codes);
        assert_eq!((scale, zero), (1.0, 127.0));
        assert_eq!(codes[0], 0);
        assert_eq!(codes[8], 64);
        assert!(codes[1..8].iter().all(|&code| code == 127));
    }
}
