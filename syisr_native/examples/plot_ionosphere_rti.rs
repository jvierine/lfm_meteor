use anyhow::{ensure, Result};
use rustplotlib::*;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 3,
        "usage: plot_ionosphere_rti PILOT.h5 OUTPUT.png"
    );
    let h = hdf5::File::open(&a[1])?;
    let power = h.dataset("block_power_sum")?.read_raw::<f64>()?;
    let counts = h.dataset("block_sample_count")?.read_raw::<u64>()?;
    let mask = h.dataset("coherent_candidate_mask")?.read_raw::<u8>()?;
    let background = h.dataset("off_echo_power_by_second")?.read_raw::<f64>()?;
    let bounds = h.dataset("utc_bounds_ns")?.read_raw::<i64>()?;
    ensure!(
        power.len() == 60 * 400 && background.len() == 60,
        "requires 60-second pilot grid"
    );
    let mut values = Vec::new();
    for b in 53..94 {
        for s in 0..60 {
            let i = s * 400 + b;
            values.push(if counts[i] > 0 && mask[i] == 0 && background[s] > 0. {
                100. * (power[i] / counts[i] as f64 / background[s] - 1.)
            } else {
                f64::NAN
            });
        }
    }
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 160.;
    f.height_mm = 95.;
    let scale = f.scale(Scale::new(
        [-5., 5.],
        Normalization::Linear,
        ColorMap::diverging(),
    )?);
    let start = chrono::DateTime::from_timestamp_nanos(bounds[0]);
    let mut p = Panel::new(
        Rect::new(23., 8., 106., 68.),
        Axes::new(
            [0., 60.],
            [79.5, 141.],
            format!("Seconds after {} UTC", start.format("%H:%M:%S")),
            "Slant range (km)",
        ),
    );
    p.push(Layer::Pcolormesh(Field::new(
        (0..=60).map(|s| s as f64).collect(),
        (53..=94).map(|b| b as f64 * 1.5).collect(),
        values,
        scale,
    )?));
    f.panels.push(p);
    f.colorbars.push(Colorbar::new(
        Rect::new(132., 8., 3., 68.),
        scale,
        r"Excess power (\%)",
        vec![-5., 0., 5.],
    ));
    f.save(&a[2])?;
    Ok(())
}
