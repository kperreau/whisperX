use pyo3::prelude::*;

mod alignment;
mod quantization;

#[pymodule]
fn whisperx_ext(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let alignment_mod = PyModule::new(py, "alignment")?;
    alignment::register(&alignment_mod)?;
    m.add_submodule(&alignment_mod)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("whisperx_ext.alignment", alignment_mod)?;
    let quantization_mod = PyModule::new(py, "quantization")?;
    quantization::register(&quantization_mod)?;
    m.add_submodule(&quantization_mod)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("whisperx_ext.quantization", quantization_mod)?;
    Ok(())
}
