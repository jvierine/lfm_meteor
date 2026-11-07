//! Native scientific figure builders. No Matplotlib scene or Python layout input.
#[path = "../native/beat.rs"]
mod beat;
#[path = "../native/common.rs"]
mod common;
#[path = "../native/detection.rs"]
mod detection;
#[path = "../native/event.rs"]
mod event;
#[path = "../native/review_event.rs"]
mod review_event;
#[path = "../native/height.rs"]
mod height;
#[path = "../native/geometry.rs"]
mod geometry;
#[path = "../native/mass.rs"]
mod mass;
#[path = "../native/detectability.rs"]
mod detectability;
#[path = "../native/ablation_detectability.rs"]
mod ablation_detectability;
#[path = "../native/rcs_mass_fit.rs"]
mod rcs_mass_fit;
#[path = "../native/rcs_integral.rs"]
mod rcs_integral;
#[path = "../native/tarnecki_lookup.rs"]
mod tarnecki_lookup;
#[path = "../native/sugar_benchmark.rs"]
mod sugar_benchmark;
#[path = "../native/orbits.rs"]
mod orbits;
#[path = "../native/radiants.rs"]
mod radiants;
#[path = "../native/rcs.rs"]
mod rcs;
use anyhow::{Result, bail};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 4,
        "usage: native-paper-figures beat|radiants scientific-data.h5 output-base"
    );
    let data = hdf5::File::open(&args[2])?;
    anyhow::ensure!(
        !data.link_exists("paths") && !data.link_exists("labels"),
        "Native figures require scientific data, not an exported graphics scene"
    );
    let mut figure = match args[1].as_str() {
        "beat" => beat::figure(&data)?,
        "radiants" => radiants::figure(&data)?,
        "mass" => mass::figure(&data)?,
        "memo-do" => detectability::figure(&data)?,
        "memo-do-both" => detectability::comparison_figure(&data)?,
        "memo-ablation" => ablation_detectability::figure(&data)?,
        "memo-rcs-mass" => rcs_mass_fit::figure(&data)?,
        "memo-rcs-examples" => rcs_mass_fit::examples(&data)?,
        "memo-rcs-density" => rcs_mass_fit::density_comparison(&data)?,
        "memo-rcs-dynamic" => rcs_mass_fit::mass_comparison(&data)?,
        "memo-rcs-floor" => rcs_mass_fit::mass_floor(&data)?,
        "memo-rcs-integral-summary" => rcs_integral::summary(&data)?,
        "memo-rcs-integral-event" => rcs_integral::event(&data)?,
        "memo-tarnecki-lookup" => tarnecki_lookup::figure(&data)?,
        "memo-sugar-benchmark" => sugar_benchmark::figure(&data)?,
        "orbits" => orbits::figure(&data)?,
        "rcs" => rcs::figure(&data)?,
        "height" => height::figure(&data)?,
        "event" => event::figure(&data)?,
        "memo-review-event" => review_event::figure(&data)?,
        "detection" => detection::figure(&data)?,
        "geometry" => geometry::figure(&data)?,
        other => bail!("No native builder for {other}; refusing a scene-converter fallback"),
    };
    let output = Path::new(&args[3]);
    std::fs::create_dir_all(output.parent().unwrap())?;
    figure.latex.cache = output.parent().unwrap().join(".native-latex-cache");
    if args[1].starts_with("memo-") {
        for extension in ["pdf", "png", "svg"] {
            figure.save(output.with_extension(extension))?;
        }
        println!("Native memo grid: {}", args[3]);
        return Ok(());
    }
    figure.save_for_article(output.with_extension("pdf"),
        &format!("figures/rustplotlib/{}.pdf", output.file_name().unwrap().to_str().unwrap()))?;
    println!(
        "Native Rust figure: {} (scientific HDF5 input: {})",
        args[3], args[2]
    );
    Ok(())
}
