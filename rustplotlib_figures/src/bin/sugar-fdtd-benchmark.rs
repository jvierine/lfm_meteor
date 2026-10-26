#[path = "../native/common.rs"]
mod common;
#[path = "../native/sugar_benchmark.rs"]
mod sugar_benchmark;

use anyhow::Result;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len() == 3, "usage: sugar-fdtd-benchmark input.h5 output-base");
    let data = hdf5::File::open(&args[1])?;
    let mut figure = sugar_benchmark::figure(&data)?;
    let output = Path::new(&args[2]);
    std::fs::create_dir_all(output.parent().unwrap())?;
    figure.latex.cache = output.parent().unwrap().join(".native-latex-cache");
    for extension in ["pdf", "png", "svg"] {
        figure.save(output.with_extension(extension))?;
    }
    Ok(())
}
