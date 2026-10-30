use pyo3::prelude::*;

mod alignment;
mod quant;

#[pymodule]
fn whisperx_ext(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let alignment_mod = PyModule::new(py, "alignment")?;
    alignment::register(&alignment_mod)?;
    m.add_submodule(&alignment_mod)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("whisperx_ext.alignment", alignment_mod)?;
    let quant_mod = PyModule::new(py, "quant")?;
    quant::register(&quant_mod)?;
    m.add_submodule(&quant_mod)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("whisperx_ext.quant", quant_mod)?;
    Ok(())
}
