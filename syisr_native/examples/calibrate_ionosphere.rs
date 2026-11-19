use anyhow::{ensure, Result};
use rustplotlib::*;
use std::path::Path;
fn median(values: &[f64]) -> f64 {
    let mut v: Vec<_> = values.iter().filter(|v| v.is_finite()).copied().collect();
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        f64::NAN
    } else {
        (v[v.len() / 2] + v[(v.len() - 1) / 2]) / 2.
    }
}
fn robust_mean(values: &[f64]) -> f64 {
    let center = median(values);
    let d: Vec<_> = values.iter().map(|v| (v - center).abs()).collect();
    let limit = 4.5 * (1.4826 * median(&d)).max(0.005 * center.abs());
    let good: Vec<_> = values
        .iter()
        .filter(|v| v.is_finite() && (**v - center).abs() <= limit)
        .collect();
    if good.is_empty() {
        f64::NAN
    } else {
        good.iter().map(|v| **v).sum::<f64>() / good.len() as f64
    }
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 2,
        "usage: calibrate_ionosphere FULL_PRODUCT_DIRECTORY"
    );
    let root = Path::new(&a[1]);
    let input = hdf5::File::with_options()
        .with_fapl(|p| p.core_filebacked(false))
        .open(root.join("rti.h5"))?;
    let times = input.dataset("window_start_utc_ns")?.read_raw::<i64>()?;
    let ranges = input.dataset("slant_range_km")?.read_raw::<f64>()?;
    let pointing = input
        .dataset("tx_azimuth_elevation_deg")?
        .read_raw::<f64>()?;
    let original = input.dataset("power_over_noise")?.read_raw::<f64>()?;
    let nt = times.len();
    let nr = ranges.len();
    let mut result = vec![f64::NAN; nt * nr];
    let mut factors = vec![f64::NAN; nt * 60];
    let mut heldout = factors.clone();
    let mut robust_noise = factors.clone();
    let mut masked = vec![0u8; nt * 60];
    for m in 0..nt {
        let path = root.join(format!("minutes/minute_{m:04}.h5"));
        if !path.exists() {
            continue;
        }
        let h = hdf5::File::with_options()
            .with_fapl(|p| p.core_filebacked(false))
            .open(path)?;
        let p = h.dataset("block_power_sum")?.read_raw::<f64>()?;
        let n = h.dataset("block_sample_count")?.read_raw::<u64>()?;
        let mask = h.dataset("coherent_candidate_mask")?.read_raw::<u8>()?;
        let predicted = h
            .dataset("robust_spectral_noise_power_by_second")?
            .read_raw::<f64>()?;
        let bounds = h
            .dataset("late_gate_noise_delay_bounds_us")?
            .read_raw::<f64>()?;
        let tail: Vec<_> = ranges
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                **r - 1. >= bounds[0] * 0.149896229 && **r + 1. <= bounds[1] * 0.149896229
            })
            .map(|(i, _)| i)
            .collect();
        for s in 0..60 {
            let ratio = |b: usize| {
                let j = s * nr + b;
                if n[j] > 0 && mask[j] == 0 && predicted[s] > 0. {
                    p[j] / n[j] as f64 / predicted[s]
                } else {
                    f64::NAN
                }
            };
            let calibration: Vec<_> = tail
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 2 == 0)
                .map(|(_, b)| ratio(*b))
                .filter(|v| v.is_finite())
                .collect();
            let validation: Vec<_> = tail
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 2 == 1)
                .map(|(_, b)| ratio(*b))
                .filter(|v| v.is_finite())
                .collect();
            if calibration.len() < 4 || validation.len() < 3 {
                masked[m * 60 + s] = 1;
                continue;
            }
            let factor = robust_mean(&calibration);
            let check = robust_mean(&validation) / factor;
            factors[m * 60 + s] = factor;
            heldout[m * 60 + s] = check;
            // Do not use tail calibration when the held-out gates disagree.
            if !check.is_finite() || (check - 1.).abs() > 0.05 {
                masked[m * 60 + s] = 2;
                continue;
            }
            robust_noise[m * 60 + s] = predicted[s] * factor;
        }
        for b in 0..nr {
            let mut power = 0.;
            let mut noise = 0.;
            for s in 0..60 {
                let j = s * nr + b;
                let q = robust_noise[m * 60 + s];
                if mask[j] == 0 && n[j] > 0 && q.is_finite() {
                    power += p[j];
                    noise += n[j] as f64 * q;
                }
            }
            if noise > 0. {
                result[m * nr + b] = power / noise;
            }
        }
        if m % 100 == 0 {
            println!("calibrated {m}/{nt} minutes");
        }
    }
    let out = hdf5::File::create(root.join("rti_noise_calibrated.h5"))?;
    for (name, v) in [
        ("power_over_noise", &result),
        ("original_power_over_noise", &original),
    ] {
        out.new_dataset::<f64>()
            .shape([nt, nr])
            .create(name)?
            .write_raw(v)?;
    }
    for (name, v) in [
        ("spectral_to_measured_noise_factor", &factors),
        ("heldout_tail_power_over_noise", &heldout),
        ("calibrated_noise_adc2", &robust_noise),
    ] {
        out.new_dataset::<f64>()
            .shape([nt, 60])
            .create(name)?
            .write_raw(v)?;
    }
    out.new_dataset::<u8>()
        .shape([nt, 60])
        .create("noise_quality_mask")?
        .write_raw(&masked)?;
    out.new_dataset_builder()
        .with_data(&times)
        .create("window_start_utc_ns")?;
    out.new_dataset_builder()
        .with_data(&ranges)
        .create("slant_range_km")?;
    out.new_dataset_builder()
        .with_data(&pointing)
        .create("tx_azimuth_elevation_deg")?;
    out.new_dataset_builder()
        .with_data(&input.dataset("complete")?.read_raw::<u8>()?)
        .create("complete")?;
    out.new_dataset_builder().with_data(b"Robust per-second empirical calibration of spectral compressed-noise prediction, using alternating 2 km late-range bins fully within saved late-delay bounds. Median/MAD clipping at 4.5 sigma, floor 0.5% of ratio. Other alternating bins are held out; reject seconds when held-out background differs by >5%. No F-region or E-region signal is used for calibration. Native masks/counts retained in original minute files. No interpolation or smoothing of RTI. This validates noise normalization, not external calibration or absence of range-dependent RFI.").create("methods_utf8")?;
    let altitude = |r: f64| {
        (6371f64.powi(2) + r * r + 2. * 6371. * r * pointing[1].to_radians().sin()).sqrt() - 6371.
    };
    let lo = ranges
        .iter()
        .position(|r| altitude(*r + 1.) >= 80.)
        .unwrap();
    let hi = ranges
        .iter()
        .position(|r| altitude(*r - 1.) >= 600.)
        .unwrap();
    let midnight = times[0].div_euclid(86400000000000) * 86400000000000;
    let t0 = (times[0] - midnight) as f64 / 3.6e12;
    let t1 = t0 + nt as f64 / 60.;
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 170.;
    fig.height_mm = 100.;
    let scale = fig.scale(Scale::new(
        [0.95, 1.5],
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
    let field = (lo..hi)
        .flat_map(|b| {
            let data = &result;
            (0..nt).map(move |m| data[m * nr + b])
        })
        .collect();
    panel.push(Layer::Pcolormesh(Field::new(
        (0..=nt).map(|m| t0 + m as f64 / 60.).collect(),
        (lo..=hi).map(|b| altitude(ranges[b] - 1.)).collect(),
        field,
        scale,
    )?));
    fig.panels.push(panel);
    fig.colorbars.push(Colorbar::new(
        Rect::new(142., 8., 3., 72.),
        scale,
        "Power / noise",
        vec![1., 1.2, 1.4],
    ));
    fig.save(root.join("rti_noise_calibrated.png"))?;
    println!(
        "Done: {nt} minutes; held-out median {:.5}; rejected {} second windows",
        median(&heldout),
        masked.iter().filter(|m| **m != 0).count()
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn correction_preserves_a_real_signal_layer() {
        let mut ratios = vec![1.65; 50];
        ratios.extend([8., 12.]);
        let factor = robust_mean(&ratios);
        assert!((factor - 1.65).abs() < 1e-12);
        let noise_prediction = 100. / 1.65;
        let signal_plus_noise = 150.;
        assert!((signal_plus_noise / (factor * noise_prediction) - 1.5).abs() < 1e-12);
    }
}
