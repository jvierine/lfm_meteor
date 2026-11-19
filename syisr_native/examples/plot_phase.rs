use anyhow::{ensure, Result};
use rustplotlib::*;
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 2, "usage: plot_phase IONOSPHERE_DIRECTORY");
    let dir = Path::new(&a[1]);
    let summary = hdf5::File::with_options()
        .with_fapl(|p| p.core_filebacked(false))
        .open(dir.join("rti.h5"))?;
    let times = summary.dataset("window_start_utc_ns")?.read_raw::<i64>()?;
    let ranges = summary.dataset("slant_range_km")?.read_raw::<f64>()?;
    let pointing = summary
        .dataset("tx_azimuth_elevation_deg")?
        .read_raw::<f64>()?;
    let nt = times.len();
    let nr = ranges.len();
    let mut phase = vec![f64::NAN; nt * nr];
    let mut coherence = phase.clone();
    for m in 0..nt {
        let path = dir.join(format!("minutes/minute_{m:04}.h5"));
        if !path.exists() {
            continue;
        }
        let h = hdf5::File::with_options()
            .with_fapl(|p| p.core_filebacked(false))
            .open(path)?;
        let p = h.dataset("lag1_phase_rad")?.read_raw::<f64>()?;
        let c = h.dataset("lag1_coherence")?.read_raw::<f64>()?;
        phase[m * nr..(m + 1) * nr].copy_from_slice(&p);
        coherence[m * nr..(m + 1) * nr].copy_from_slice(&c);
    }
    let out = hdf5::File::create(dir.join("lag1_phase_snapshot.h5"))?;
    for (name, v) in [("lag1_phase_rad", &phase), ("lag1_coherence", &coherence)] {
        out.new_dataset::<f64>()
            .shape([nt, nr])
            .create(name)?
            .write_raw(v)?;
    }
    out.new_dataset_builder()
        .with_data(&times)
        .create("window_start_utc_ns")?;
    out.new_dataset_builder()
        .with_data(&ranges)
        .create("slant_range_km")?;
    out.new_dataset_builder()
        .with_data(&pointing)
        .create("tx_azimuth_elevation_deg")?;
    out.new_dataset_builder().with_data(b"Phase=arg(sum y_next conj(y_previous)) for accepted 5 ms pairs, 60-second windows. No phase unwrapping, branch selection, or significance masking; low-coherence phase is not a measured plasma velocity.").create("methods_utf8")?;
    let altitude = |r: f64| {
        (6371f64.powi(2) + r * r + 2. * 6371. * r * pointing[1].to_radians().sin()).sqrt() - 6371.
    };
    let midnight = times[0].div_euclid(86400000000000) * 86400000000000;
    let t0 = (times[0] - midnight) as f64 / 3.6e12;
    let t1 = t0 + nt as f64 / 60.;
    for (name, ybounds) in [
        ("lag1_phase_full", [80., 600.]),
        ("lag1_phase_e_region", [80., 150.]),
    ] {
        let lo = ranges
            .iter()
            .position(|r| altitude(*r + 1.) >= ybounds[0])
            .unwrap();
        let hi = ranges
            .iter()
            .position(|r| altitude(*r - 1.) >= ybounds[1])
            .unwrap();
        let mut fig = Figure::new(IntendedUse::ArticleWide);
        fig.width_mm = 170.;
        fig.height_mm = 157.;
        let cyclic = ColorMap::new(vec![
            [0.3, 0.1, 0.5],
            [0.1, 0.5, 0.8],
            [0.2, 0.8, 0.4],
            [1., 0.9, 0.3],
            [0.9, 0.2, 0.1],
            [0.3, 0.1, 0.5],
        ])?;
        for (row, (values, label, limits, colors)) in [
            (
                &phase,
                "Lag-1 phase (rad)",
                [-std::f64::consts::PI, std::f64::consts::PI],
                cyclic,
            ),
            (&coherence, "Lag-1 coherence", [0., 0.01], ColorMap::storm()),
        ]
        .into_iter()
        .enumerate()
        {
            let scale = fig.scale(Scale::new(limits, Normalization::Linear, colors)?);
            let y = 8. + row as f64 * 72.;
            let mut p = Panel::new(
                Rect::new(23., y, 116., 54.),
                Axes::new([t0, t1], ybounds, "UTC hour (2026-11-17)", "Altitude (km)"),
            );
            let v = (lo..hi)
                .flat_map(|b| (0..nt).map(move |m| values[m * nr + b]))
                .collect();
            p.push(Layer::Pcolormesh(Field::new(
                (0..=nt).map(|m| t0 + m as f64 / 60.).collect(),
                (lo..=hi).map(|b| altitude(ranges[b] - 1.)).collect(),
                v,
                scale,
            )?));
            fig.panels.push(p);
            fig.colorbars.push(Colorbar::new(
                Rect::new(142., y, 3., 54.),
                scale,
                label,
                if row == 0 {
                    vec![-3., 0., 3.]
                } else {
                    vec![0., 0.005, 0.01]
                },
            ));
        }
        fig.save(dir.join(format!("{name}.png")))?;
    }
    println!("Published phase and coherence for {nt} completed minutes");
    Ok(())
}
