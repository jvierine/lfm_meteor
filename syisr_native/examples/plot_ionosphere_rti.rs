use anyhow::{ensure, Result};
use rustplotlib::*;
fn altitude_km(r: f64, elevation_deg: f64) -> f64 {
    (6371f64.powi(2) + r * r + 2. * 6371. * r * elevation_deg.to_radians().sin()).sqrt() - 6371.
}
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
    let background = h
        .dataset("late_gate_noise_power_by_second")?
        .read_raw::<f64>()?;
    let ranges = h.dataset("slant_range_km")?.read_raw::<f64>()?;
    let pointing = h.dataset("tx_azimuth_elevation_deg")?.read_raw::<f64>()?;
    let altitude = |r: f64| altitude_km(r, pointing[1]);
    let dr = ranges[1] - ranges[0];
    let bounds = h.dataset("utc_bounds_ns")?.read_raw::<i64>()?;
    ensure!(
        power.len() == 60 * 400 && background.len() == 60,
        "requires 60-second pilot grid"
    );
    let mut values = Vec::new();
    let lo = ranges
        .iter()
        .position(|r| altitude(*r + dr / 2.) >= 80.)
        .unwrap();
    let hi = ranges
        .iter()
        .position(|r| altitude(*r - dr / 2.) >= 600.)
        .unwrap();
    for b in lo..hi {
        for s in 0..60 {
            let i = s * 400 + b;
            values.push(if counts[i] > 0 && mask[i] == 0 && background[s] > 0. {
                power[i] / counts[i] as f64 / background[s]
            } else {
                f64::NAN
            });
        }
    }
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 160.;
    f.height_mm = 95.;
    // Storm RGB control points, peepomap 0.2.2 (MIT). Native interpolation:
    // https://pypi.org/project/peepomap/ ; no Matplotlib runtime is used.
    let storm = ColorMap::new(vec![
        [0.2, 0.2, 0.7],
        [0.4, 0.3, 0.9],
        [0.5, 0.5, 1.0],
        [0.7, 0.6, 1.0],
        [0.8, 0.8, 1.0],
        [1.0, 0.9, 0.9],
        [1.0, 0.6, 0.6],
        [0.9, 0.4, 0.4],
        [0.8, 0.2, 0.2],
        [0.6, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [1.0, 0.2, 0.0],
        [1.0, 0.4, 0.0],
        [1.0, 0.6, 0.0],
        [1.0, 0.8, 0.2],
        [1.0, 0.9, 0.5],
        [1.0, 1.0, 0.7],
        [0.9, 1.0, 1.0],
        [0.6, 0.8, 1.0],
        [0.3, 0.6, 1.0],
        [0.2, 0.5, 0.9],
        [0.1, 0.4, 0.8],
        [0.0, 0.3, 0.7],
        [0.0, 0.5, 0.0],
        [0.1, 0.7, 0.1],
        [0.3, 0.9, 0.3],
        [0.6, 1.0, 0.5],
        [0.8, 1.0, 0.7],
        [0.4, 0.4, 0.4],
        [0.6, 0.6, 0.6],
        [0.7, 0.7, 0.7],
        [0.9, 0.9, 0.9],
    ])?;
    let scale = f.scale(Scale::new([0.95, 1.5], Normalization::Linear, storm)?);
    let start = chrono::DateTime::from_timestamp_nanos(bounds[0]);
    let mut p = Panel::new(
        Rect::new(23., 8., 106., 68.),
        Axes::new(
            [0., 60.],
            [80., 600.],
            format!("Seconds after {} UTC", start.format("%H:%M:%S")),
            "Altitude (km)",
        ),
    );
    p.push(Layer::Pcolormesh(Field::new(
        (0..=60).map(|s| s as f64).collect(),
        (lo..=hi).map(|b| altitude(ranges[b] - dr / 2.)).collect(),
        values.clone(),
        scale,
    )?));
    f.panels.push(p);
    f.colorbars.push(Colorbar::new(
        Rect::new(132., 8., 3., 68.),
        scale,
        "Power / noise",
        vec![1., 1.2, 1.4],
    ));
    f.save(&a[2])?;
    let product = hdf5::File::create(std::path::Path::new(&a[2]).with_extension("h5"))?;
    product
        .new_dataset::<f64>()
        .shape([hi - lo, 60])
        .create("power_over_noise")?
        .write_raw(&values)?;
    product
        .new_dataset_builder()
        .with_data(
            &(lo..=hi)
                .map(|b| altitude(ranges[b] - dr / 2.))
                .collect::<Vec<_>>(),
        )
        .create("altitude_edges_km")?;
    product
        .new_dataset_builder()
        .with_data(&background)
        .create("noise_power_by_second_adc2")?;
    product
        .new_dataset_builder()
        .with_data(&bounds)
        .create("utc_bounds_ns")?;
    product.new_dataset_builder().with_data(b"Straight ray at recorded elevation; spherical Earth radius 6371 km; site height assumed zero. Approximate geometric altitude, no refraction. Power/noise includes noise; no subtraction. Noise from latest full-pulse matched-filter gates available near end of IPP. Storm control points from peepomap 0.2.2.").create("methods_utf8")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn range_is_not_altitude() {
        assert!((altitude_km(600., 90.) - 600.).abs() < 1e-9);
        assert!(altitude_km(600., 75.) < 600.);
        assert!(altitude_km(620., 75.) > 600.);
        assert!(altitude_km(102., 75.) < 102.);
    }
}
