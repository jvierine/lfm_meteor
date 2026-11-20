use anyhow::{ensure, Result};
use rustplotlib::*;
use std::path::Path;
fn db(p: f64) -> f64 {
    if p.is_finite() && p > 0. {
        10. * p.log10()
    } else {
        f64::NAN
    }
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "usage: plot_power_db RTI.h5 OUTPUT.png");
    let h = hdf5::File::open(&a[1])?;
    let p = h.dataset("power_over_noise")?.read_raw::<f64>()?;
    let t = h.dataset("window_start_utc_ns")?.read_raw::<i64>()?;
    let r = h.dataset("slant_range_km")?.read_raw::<f64>()?;
    let el = h.dataset("tx_azimuth_elevation_deg")?.read_raw::<f64>()?[1];
    let altitude =
        |r: f64| (6371f64.powi(2) + r * r + 2. * 6371. * r * el.to_radians().sin()).sqrt() - 6371.;
    let lo = r.iter().position(|r| altitude(*r + 1.) >= 80.).unwrap();
    let hi = r.iter().position(|r| altitude(*r - 1.) >= 600.).unwrap();
    let midnight = t[0].div_euclid(86400000000000) * 86400000000000;
    let t0 = (t[0] - midnight) as f64 / 3.6e12;
    let t1 = t0 + t.len() as f64 / 60.;
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 170.;
    f.height_mm = 100.;
    let scale = f.scale(Scale::new(
        [-0.2, 2.],
        Normalization::Linear,
        ColorMap::storm(),
    )?);
    let mut panel = Panel::new(
        Rect::new(23., 8., 116., 72.),
        Axes::new(
            [t0, t1],
            [80., 600.],
            "UTC hour (2026-11-17)",
            "Altitude (km)",
        ),
    );
    let nr = r.len();
    let values: Vec<_> = (lo..hi)
        .flat_map(|b| {
            let p = &p;
            (0..t.len()).map(move |i| db(p[i * nr + b]))
        })
        .collect();
    panel.push(Layer::Pcolormesh(Field::new(
        (0..=t.len()).map(|i| t0 + i as f64 / 60.).collect(),
        (lo..=hi).map(|b| altitude(r[b] - 1.)).collect(),
        values,
        scale,
    )?));
    f.panels.push(panel);
    f.colorbars.push(Colorbar::new(
        Rect::new(142., 8., 3., 72.),
        scale,
        "Power / noise (dB)",
        vec![0., 1., 2.],
    ));
    f.save(&a[2])?;
    let out = hdf5::File::create(Path::new(&a[2]).with_extension("h5"))?;
    out.new_dataset::<f64>()
        .shape([t.len(), r.len()])
        .create("power_over_noise_db")?
        .write_raw(&p.iter().map(|p| db(*p)).collect::<Vec<_>>())?;
    out.new_dataset_builder()
        .with_data(&t)
        .create("window_start_utc_ns")?;
    out.new_dataset_builder()
        .with_data(&r)
        .create("slant_range_km")?;
    out.new_dataset_builder().with_data(b"10 log10(total power / calibrated noise), not 10 log10(P/N-1). Negative dB estimates retained. No extra time/range smoothing.").create("conventions_utf8")?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decibel_conversion_keeps_negative_estimates() {
        assert_eq!(db(1.), 0.);
        assert!((db(10.) - 10.).abs() < 1e-12);
        assert!(db(0.9) < 0.);
        assert!(db(0.).is_nan());
    }
}
