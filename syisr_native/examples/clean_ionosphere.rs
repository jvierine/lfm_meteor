use anyhow::{ensure, Result};
use num_complex::Complex64;
use rustplotlib::*;
use std::path::Path;
fn median(v: &[f64]) -> f64 {
    let mut x: Vec<_> = v.iter().filter(|x| x.is_finite()).copied().collect();
    x.sort_by(f64::total_cmp);
    if x.is_empty() {
        f64::NAN
    } else {
        (x[x.len() / 2] + x[(x.len() - 1) / 2]) / 2.
    }
}
fn robust_noise(values: &[f64]) -> f64 {
    let center = median(values);
    let deviations: Vec<_> = values.iter().map(|v| (v - center).abs()).collect();
    let sigma = 1.4826 * median(&deviations);
    let good: Vec<_> = values
        .iter()
        .filter(|v| v.is_finite() && (**v - center).abs() <= 4.5 * sigma.max(center * 0.002))
        .copied()
        .collect();
    if good.is_empty() {
        center
    } else {
        good.iter().sum::<f64>() / good.len() as f64
    }
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 2,
        "usage: clean_ionosphere FULL_PRODUCT_DIRECTORY"
    );
    let dest = Path::new(&a[1]);
    let summary = hdf5::File::open(dest.join("rti.h5"))?;
    let times = summary.dataset("window_start_utc_ns")?.read_raw::<i64>()?;
    let ranges = summary.dataset("slant_range_km")?.read_raw::<f64>()?;
    let pointing = summary
        .dataset("tx_azimuth_elevation_deg")?
        .read_raw::<f64>()?;
    let nr = ranges.len();
    let nt = times.len();
    let mut power_rows = vec![f64::NAN; nt * nr];
    let mut phases = power_rows.clone();
    let mut coherences = power_rows.clone();
    let mut noise_series = vec![f64::NAN; nt];
    let mut rejections = vec![0u64; nt];
    let mut saved_mask = vec![0u8; nt * 60 * nr];
    for m in 0..nt {
        let path = dest.join(format!("minutes/minute_{m:04}.h5"));
        if !path.exists() {
            continue;
        }
        let h = hdf5::File::with_options()
            .with_fapl(|p| p.core_filebacked(false))
            .open(path)?;
        let p = h.dataset("block_power_sum")?.read_raw::<f64>()?;
        let n = h.dataset("block_sample_count")?.read_raw::<u64>()?;
        let ar = h.dataset("block_lag1_real_sum")?.read_raw::<f64>()?;
        let ai = h.dataset("block_lag1_imag_sum")?.read_raw::<f64>()?;
        let pairs = h.dataset("block_pair_count")?.read_raw::<u64>()?;
        let old = h.dataset("coherent_candidate_mask")?.read_raw::<u8>()?;
        let delay = h
            .dataset("late_gate_noise_delay_bounds_us")?
            .read_raw::<f64>()?;
        let rlo = delay[0] * 0.149896229;
        let rhi = delay[1] * 0.149896229;
        let tail: Vec<_> = (0..60)
            .flat_map(|s| {
                ranges
                    .iter()
                    .enumerate()
                    .filter(move |(_, r)| **r >= rlo && **r <= rhi)
                    .map(move |(b, _)| s * nr + b)
            })
            .filter(|j| n[*j] > 0 && old[*j] == 0)
            .map(|j| p[j] / n[j] as f64)
            .collect();
        let noise = robust_noise(&tail);
        noise_series[m] = noise;
        let profiles: Vec<_> = p
            .iter()
            .zip(&n)
            .map(|(p, n)| if *n > 0 { p / *n as f64 } else { f64::NAN })
            .collect();
        let mut candidate = vec![false; 60 * nr];
        let mut strong = candidate.clone();
        for s in 0..60 {
            for b in 1..nr - 1 {
                let j = s * nr + b;
                if n[j] == 0 {
                    continue;
                }
                let lo = b.saturating_sub(8);
                let hi = (b + 9).min(nr);
                let local = median(&profiles[s * nr + lo..s * nr + hi]);
                let deviations: Vec<_> = profiles[s * nr + lo..s * nr + hi]
                    .iter()
                    .map(|p| (p - local).abs())
                    .collect();
                let threshold = (0.05 * noise).max(4.5 * 1.4826 * median(&deviations));
                candidate[j] = profiles[j] > local + threshold;
                strong[j] = profiles[j] > local + 0.25 * noise;
            }
        }
        let mut reject: Vec<_> = old.iter().map(|v| *v != 0).collect();
        for s in 0..60 {
            for b in 0..nr {
                if !candidate[s * nr + b] {
                    continue;
                }
                let track = [
                    s.checked_sub(1),
                    if s + 1 < 60 { Some(s + 1) } else { None },
                ]
                .into_iter()
                .flatten()
                .any(|t| (b.saturating_sub(5)..(b + 6).min(nr)).any(|k| candidate[t * nr + k]));
                if track || strong[s * nr + b] {
                    for t in s.saturating_sub(1)..(s + 2).min(60) {
                        for k in b.saturating_sub(2)..(b + 3).min(nr) {
                            reject[t * nr + k] = true;
                        }
                    }
                }
            }
        }
        rejections[m] = reject.iter().filter(|v| **v).count() as u64;
        for (j, v) in reject.iter().enumerate() {
            saved_mask[m * 60 * nr + j] = u8::from(*v);
        }
        for b in 0..nr {
            let mut total = 0.;
            let mut count = 0u64;
            let mut ac = Complex64::default();
            let mut np = 0u64;
            for s in 0..60 {
                let j = s * nr + b;
                if !reject[j] {
                    total += p[j];
                    count += n[j];
                    ac += Complex64::new(ar[j], ai[j]);
                    np += pairs[j];
                }
            }
            let j = m * nr + b;
            if count > 0 {
                power_rows[j] = total / count as f64 / noise;
            }
            if np > 0 && total > 0. {
                phases[j] = ac.arg();
                coherences[j] = ac.norm() / np as f64 / (total / count as f64);
            }
        }
    }
    let quiet = |lo: f64, hi: f64| {
        (0..nt)
            .map(|m| {
                median(
                    &ranges
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| **r >= lo && **r < hi)
                        .map(|(b, _)| power_rows[m * nr + b])
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    };
    let qlow = quiet(120., 180.);
    let qhigh = quiet(550., 600.);
    let low_center = median(&qlow);
    let high_center = median(&qhigh);
    let mut bad_columns = vec![0u8; nt];
    let mut narrow_mask = vec![0u8; nt * nr];
    for m in 0..nt {
        let dl = qlow[m] - low_center;
        let dh = qhigh[m] - high_center;
        if dl.abs() > 0.018 && dh.abs() > 0.018 && dl * dh > 0. {
            bad_columns[m] = 1;
            for b in 0..nr {
                power_rows[m * nr + b] = f64::NAN;
                phases[m * nr + b] = f64::NAN;
                coherences[m * nr + b] = f64::NAN;
            }
            continue;
        }
        for b in 0..nr {
            let local = median(&power_rows[m * nr + b.saturating_sub(8)..m * nr + (b + 9).min(nr)]);
            if power_rows[m * nr + b] > local + 0.025 {
                for k in b.saturating_sub(2)..(b + 3).min(nr) {
                    narrow_mask[m * nr + k] = 1;
                }
            }
        }
    }
    for (j, mask) in narrow_mask.iter().enumerate() {
        if *mask != 0 {
            power_rows[j] = f64::NAN;
            phases[j] = f64::NAN;
            coherences[j] = f64::NAN;
        }
    }
    let out = hdf5::File::create(dest.join("rti_cleaned.h5"))?;
    for (name, v) in [
        ("power_over_noise", &power_rows),
        ("lag1_phase_rad", &phases),
        ("lag1_coherence", &coherences),
    ] {
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
        .with_data(&noise_series)
        .create("robust_late_gate_noise_adc2")?;
    out.new_dataset_builder()
        .with_data(&rejections)
        .create("rejected_second_range_cells")?;
    out.new_dataset_builder()
        .with_data(&bad_columns)
        .create("common_background_shift_time_mask")?;
    out.new_dataset::<u8>()
        .shape([nt, nr])
        .create("post_average_narrow_mask")?
        .write_raw(&narrow_mask)?;
    out.new_dataset::<u8>()
        .shape([nt, 60, nr])
        .create("scatter_rejection_mask")?
        .write_raw(&saved_mask)?;
    out.new_dataset_builder().with_data(b"Robust late-gate noise from one-second/range-cell means: median/MAD 4.5 sigma clipping then mean. Scatter candidates exceed 5% noise and 4.5 local MAD above 34 km local median. Require adjacent-second support within 10 km, or excess >25% noise. Dilate +/-1 second and +/-4 km. Heuristic satellite/coherent/RFI rejection, not object identification. Phase is raw lag1, not de-aliased plasma velocity.").create("methods_utf8")?;
    out.new_dataset_builder().with_data(b"Additional review masks: time column rejected if both 120-180 km and 550-600 km quiet-range median power/noise shift >0.018 from their experiment medians with same sign. Residual narrow range peaks >0.025 above 34 km local median are masked +/-4 km. Conservative flags may exclude real narrow layers. No interpolation or replacement; masks are saved. Full raw spectral whitening has NOT been applied to these aggregate products.").create("additional_mask_method_utf8")?;
    let altitude = |r: f64| {
        (6371f64.powi(2) + r * r + 2. * 6371. * r * pointing[1].to_radians().sin()).sqrt() - 6371.
    };
    let midnight = times[0].div_euclid(86400000000000) * 86400000000000;
    let t0 = (times[0] - midnight) as f64 / 3.6e12;
    let t1 = t0 + nt as f64 / 60.;
    for (name, ybounds, fields) in [
        (
            "rti_cleaned",
            [80., 600.],
            vec![(&power_rows, "Power / noise", [0.95, 1.5], ColorMap::storm())],
        ),
        (
            "phase_full",
            [80., 600.],
            vec![
                (
                    &phases,
                    "Lag-1 phase (rad)",
                    [-0.1, 0.1],
                    ColorMap::diverging(),
                ),
                (
                    &coherences,
                    "Lag-1 coherence",
                    [0., 0.08],
                    ColorMap::storm(),
                ),
            ],
        ),
        (
            "phase_e_region",
            [80., 150.],
            vec![
                (
                    &phases,
                    "Lag-1 phase (rad)",
                    [-0.1, 0.1],
                    ColorMap::diverging(),
                ),
                (
                    &coherences,
                    "Lag-1 coherence",
                    [0., 0.08],
                    ColorMap::storm(),
                ),
            ],
        ),
    ] {
        let lo = ranges
            .iter()
            .position(|r| altitude(*r + 1.) >= ybounds[0])
            .unwrap();
        let hi = ranges
            .iter()
            .position(|r| altitude(*r - 1.) >= ybounds[1])
            .unwrap();
        let mut f = Figure::new(IntendedUse::ArticleWide);
        f.width_mm = 170.;
        f.height_mm = if fields.len() == 1 { 95. } else { 170. };
        for (row, (values, label, limits, colors)) in fields.into_iter().enumerate() {
            let scale = f.scale(Scale::new(limits, Normalization::Linear, colors)?);
            let mut p = Panel::new(
                Rect::new(23., 8. + row as f64 * 78., 116., 58.),
                Axes::new([t0, t1], ybounds, "UTC hour (2026-11-17)", "Altitude (km)"),
            );
            let v: Vec<_> = (lo..hi)
                .flat_map(|b| (0..nt).map(move |t| values[t * nr + b]))
                .collect();
            p.push(Layer::Pcolormesh(Field::new(
                (0..=nt).map(|t| t0 + t as f64 / 60.).collect(),
                (lo..=hi).map(|b| altitude(ranges[b] - 1.)).collect(),
                v,
                scale,
            )?));
            f.panels.push(p);
            f.colorbars.push(Colorbar::new(
                Rect::new(142., 8. + row as f64 * 78., 3., 58.),
                scale,
                label,
                vec![limits[0], (limits[0] + limits[1]) / 2., limits[1]],
            ));
        }
        f.save(dest.join(format!("{name}.png")))?;
    }
    println!("Published {nt} cleaned minutes and full/E-region phase plots");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn robust_noise_rejects_positive_contamination() {
        let mut v = vec![100.; 100];
        v.extend(vec![1000.; 10]);
        assert_eq!(robust_noise(&v), 100.);
    }
}
