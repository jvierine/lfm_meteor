use anyhow::{ensure, Result};
use rustplotlib::*;
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 3,
        "usage: plot_linear_colormaps RTI.h5 OUTPUT_DIRECTORY"
    );
    let h = hdf5::File::open(&a[1])?;
    let p = h.dataset("power_over_noise")?.read_raw::<f64>()?;
    let t = h.dataset("window_start_utc_ns")?.read_raw::<i64>()?;
    let r = h.dataset("slant_range_km")?.read_raw::<f64>()?;
    let el = h.dataset("tx_azimuth_elevation_deg")?.read_raw::<f64>()?[1];
    let altitude =
        |r: f64| (6371f64.powi(2) + r * r + 2. * 6371. * r * el.to_radians().sin()).sqrt() - 6371.;
    let lo = r.iter().position(|r| altitude(*r + 1.) >= 80.).unwrap();
    let hi = r.iter().position(|r| altitude(*r - 1.) >= 600.).unwrap();
    let nr = r.len();
    let midnight = t[0].div_euclid(86400000000000) * 86400000000000;
    let t0 = (t[0] - midnight) as f64 / 3.6e12;
    let t1 = t0 + t.len() as f64 / 60.;
    let values: Vec<_> = (lo..hi)
        .flat_map(|b| {
            let p = &p;
            (0..t.len()).map(move |i| p[i * nr + b])
        })
        .collect();
    for (name, colors) in [
        ("inferno", ColorMap::inferno()),
        ("plasma", ColorMap::plasma()),
        (
            "grayscale",
            ColorMap::new(vec![[0., 0., 0.], [1., 1., 1.]])?,
        ),
    ] {
        let mut fig = Figure::new(IntendedUse::ArticleWide);
        fig.width_mm = 170.;
        fig.height_mm = 100.;
        let scale = fig.scale(Scale::new([0.967, 1.03], Normalization::Linear, colors)?);
        let mut panel = Panel::new(
            Rect::new(23., 8., 116., 72.),
            Axes::new(
                [t0, t1],
                [80., 600.],
                "UTC hour (2026-11-17)",
                "Altitude (km)",
            ),
        );
        panel.push(Layer::Pcolormesh(Field::new(
            (0..=t.len()).map(|i| t0 + i as f64 / 60.).collect(),
            (lo..=hi).map(|b| altitude(r[b] - 1.)).collect(),
            values.clone(),
            scale,
        )?));
        fig.panels.push(panel);
        fig.colorbars.push(Colorbar::new(
            Rect::new(142., 8., 3., 72.),
            scale,
            "Power / noise",
            vec![0.97, 1., 1.03],
        ));
        fig.save(Path::new(&a[2]).join(format!("full_linear_{name}.png")))?;
    }
    let out = hdf5::File::create(Path::new(&a[2]).join("linear_colormap_comparison.h5"))?;
    out.new_dataset::<f64>()
        .shape([t.len(), nr])
        .create("power_over_noise")?
        .write_raw(&p)?;
    out.new_dataset_builder()
        .with_data(&[0.967, 1.03])
        .create("display_limits")?;
    out.new_dataset_builder()
        .with_data(&t)
        .create("window_start_utc_ns")?;
    out.new_dataset_builder()
        .with_data(&r)
        .create("slant_range_km")?;
    out.new_dataset_builder()
        .with_data(&[el])
        .create("elevation_deg")?;
    out.new_dataset_builder().with_data(b"Identical original calibrated total-power/noise arrays and linear normalization [0.967,1.03] for Inferno, Plasma and grayscale. Full observing time, 80-600 km, no smoothing, clipping of stored values or interpolation. Colour limits affect display only.").create("methods_utf8")?;
    Ok(())
}
