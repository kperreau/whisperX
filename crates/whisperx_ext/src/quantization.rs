//! Fused FP32 reconstruction for the native INT8 emission engine.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[inline(always)]
fn dequantize(
    products: &[f32],
    scales: &[f32],
    zeros: &[f32],
    weight_scales: &[f32],
    weight_sums: &[i32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    let columns = weight_scales.len();
    for (row, (source, target)) in products
        .chunks_exact(columns)
        .zip(output.chunks_exact_mut(columns))
        .enumerate()
    {
        let scale = scales[row];
        let zero = zeros[row];
        for column in 0..columns {
            let corrected = source[column] - zero * weight_sums[column] as f32;
            let combined_scale = scale * weight_scales[column];
            let value = corrected * combined_scale;
            target[column] = if let Some(bias) = bias {
                value + bias[column]
            } else {
                value
            };
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_avx2(
    products: &[f32],
    scales: &[f32],
    zeros: &[f32],
    weight_scales: &[f32],
    weight_sums: &[i32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    dequantize(
        products,
        scales,
        zeros,
        weight_scales,
        weight_sums,
        bias,
        output,
    )
}

/// Reconstruct row/channel scales, zero-point correction and bias in one pass.
/// Preserve the separate FP32 rounding of the Tensor reference operations.
#[pyfunction]
#[pyo3(signature = (products, scales, zeros, weight_scales, weight_sums, bias=None))]
fn dequantize_rows<'py>(
    py: Python<'py>,
    products: PyReadonlyArray2<'py, f32>,
    scales: PyReadonlyArray2<'py, f32>,
    zeros: PyReadonlyArray2<'py, f32>,
    weight_scales: PyReadonlyArray1<'py, f32>,
    weight_sums: PyReadonlyArray1<'py, i32>,
    bias: Option<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let product = products.as_array();
    let (rows, columns) = product.dim();
    let scale_array = scales.as_array();
    let zero_array = zeros.as_array();
    let weight_scale_array = weight_scales.as_array();
    let weight_sum_array = weight_sums.as_array();
    let bias_array = bias.as_ref().map(|v| v.as_array());
    if columns == 0
        || scale_array.dim() != (rows, 1)
        || zero_array.dim() != (rows, 1)
        || weight_scale_array.len() != columns
        || weight_sum_array.len() != columns
        || bias_array.as_ref().is_some_and(|v| v.len() != columns)
    {
        return Err(PyValueError::new_err(
            "Dequantization dimensions do not match",
        ));
    }
    let input = product
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Products must be contiguous"))?;
    let scale = scale_array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Scales must be contiguous"))?;
    let zero = zero_array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Zeros must be contiguous"))?;
    let weight_scale = weight_scale_array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Weight scales must be contiguous"))?;
    let weight_sum = weight_sum_array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Weight sums must be contiguous"))?;
    let bias_slice = bias_array
        .as_ref()
        .map(|v| {
            v.as_slice()
                .ok_or_else(|| PyValueError::new_err("Bias must be contiguous"))
        })
        .transpose()?;
    let mut output = vec![0_f32; input.len()];
    #[cfg(target_arch = "x86_64")]
    {
        // The specialized function is called only after the runtime ISA check.
        if is_x86_feature_detected!("avx2") {
            unsafe {
                dequantize_avx2(
                    input,
                    scale,
                    zero,
                    weight_scale,
                    weight_sum,
                    bias_slice,
                    &mut output,
                )
            };
        } else {
            dequantize(
                input,
                scale,
                zero,
                weight_scale,
                weight_sum,
                bias_slice,
                &mut output,
            );
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        dequantize(
            input,
            scale,
            zero,
            weight_scale,
            weight_sum,
            bias_slice,
            &mut output,
        );
    }
    Ok(Array2::from_shape_vec((rows, columns), output)
        .unwrap()
        .into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dequantize_rows, m)?)?;
    Ok(())
}
