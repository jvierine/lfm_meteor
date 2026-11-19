// Pilot 60-second LFM power and pulse-pair products, using original recordings.
#[path = "../src/filter.rs"]
#[allow(dead_code)]
mod filter;
#[path = "../src/raw.rs"]
#[allow(dead_code)]
mod raw;
use anyhow::{ensure, Result};
use num_complex::{Complex32, Complex64};
use rustplotlib::*;
use std::path::Path;
#[derive(Clone, Default)]
struct Cell {
    p: f64,
    noise: f64,
    n: u64,
    ac: Complex64,
    pairs: u64,
    variance: f64,
}
fn pulse_pair_hz(ac: Complex64) -> f64 {
    ac.arg() / (std::f64::consts::TAU * 0.005)
}
fn coherent_candidate(c: &Cell) -> bool {
    c.n > 0
        && c.pairs > 0
        && c.p > 0.
        && c.ac.norm() / c.pairs as f64 / (c.p / c.n as f64) > 0.15
        && c.p / c.noise > 1.1
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 3,
        "usage: ionosphere PRODUCT_DIRECTORY OUTPUT_PREFIX"
    );
    process(Path::new(&args[1]),&args[2],0,0)?;
    Ok(())
}
pub fn process(root:&Path,prefix:&str,start_ns:i64,start_file:usize)->Result<(i64,usize,u64)> {
    let cat = hdf5::File::open(root.join("catalogue.h5"))?;
    let starts = cat.dataset("event_start_utc_ns")?.read_raw::<i64>()?;
    let stops = cat.dataset("event_stop_utc_ns")?.read_raw::<i64>()?;
    let mut epoch = start_ns;
    let mut last_file=start_file;
    let mut grid = vec![Cell::default(); 60 * 400];
    let mut late_power = vec![0.; 60];
    let mut late_count = vec![0u64; 60];
    let mut late_delay_min = f64::INFINITY;
    let mut late_delay_max = 0f64;
    let mut previous: Option<(i64, f64, Vec<Complex32>, Vec<bool>)> = None;
    let mut used = 0u64;
    let mut excluded = 0u64;
    let mut az = 0.;
    let mut el = 0.;
    'files: for file_index in start_file..854 {
        last_file=file_index;
        let index = hdf5::File::open(root.join(format!("work/pulses_{file_index:06}.h5")))?;
        let t = index.dataset("pulses/utc_ns")?.read_raw::<i64>()?;
        let offsets = index
            .dataset("pulses/frame_byte_offset")?
            .read_raw::<u64>()?;
        let path = String::from_utf8(index.dataset("source_path_utf8")?.read_raw::<u8>()?)?;
        // Reject timestamps outside the monotone main run; no invented times.
        let mut tails: Vec<usize> = Vec::new();
        let mut prior = vec![usize::MAX; t.len()];
        for (i, time) in t.iter().enumerate() {
            let k = tails.partition_point(|j| t[*j] < *time);
            if k > 0 {
                prior[i] = tails[k - 1];
            }
            if k == tails.len() {
                tails.push(i);
            } else {
                tails[k] = i;
            }
        }
        let mut valid = vec![false; t.len()];
        let mut j = *tails.last().unwrap();
        loop {
            valid[j] = true;
            if prior[j] == usize::MAX {
                break;
            }
            j = prior[j];
        }
        let mut reader = raw::Reader::open(Path::new(&path), 8, 8)?;
        for i in 0..t.len() {
            if !valid[i] {
                continue;
            }
            if epoch == 0 {
                epoch = t[i];
            }
            if t[i] < epoch {
                continue;
            }
            if t[i] >= epoch + 60_000_000_000 {
                break 'files;
            }
            if starts
                .iter()
                .zip(&stops)
                .any(|(a, b)| t[i] >= a - 100_000_000 && t[i] <= b + 100_000_000)
            {
                excluded += 1;
                previous = None;
                continue;
            }
            reader.seek(offsets[i])?;
            let fr = reader.next()?.unwrap();
            ensure!(fr.header.utc_ns == t[i], "source timestamp mismatch");
            ensure!(
                (fr.header.ipp_us - 5000.).abs() < 1.,
                "pilot requires 5 ms IPP"
            );
            az = fr.header.az_deg;
            el = fr.header.el_deg;
            let mut powers: Vec<_> = fr.iq.iter().map(|z| z.norm_sqr()).collect();
            let noise = filter::median(&mut powers) as f64 / 2f64.ln() * 800.;
            let mut f = filter::Filter::new(fr.iq.len(), true);
            let y = f.apply(&fr.iq);
            let second = ((t[i] - epoch) / 1_000_000_000) as usize;
            let mask: Vec<_> = y
                .iter()
                .map(|z| (z.norm_sqr() as f64) < 100. * noise)
                .collect();
            // Last 200 us of valid full-pulse matched-filter lags. The recorder
            // does not extend to the literal end of the 5 ms IPP.
            for j in y.len().saturating_sub(800)..y.len() {
                if mask[j] {
                    late_power[second] += y[j].norm_sqr() as f64;
                    late_count[second] += 1;
                    let delay = fr.header.gate_front_us + j as f64 / 4.;
                    late_delay_min = late_delay_min.min(delay);
                    late_delay_max = late_delay_max.max(delay);
                }
            }
            for (j, z) in y.iter().enumerate() {
                let r = filter::C / 2e9 * (fr.header.gate_front_us + j as f64 / 4.);
                let bin = (r / 2.).floor() as usize;
                if bin >= 400 || r < 80. || !mask[j] {
                    continue;
                }
                let c = &mut grid[second * 400 + bin];
                c.p += z.norm_sqr() as f64;
                c.noise += noise;
                c.n += 1;
                if let Some((pt, pg, prev, pm)) = &previous {
                    if t[i] - pt == 5_000_000
                        && *pg == fr.header.gate_front_us
                        && j < prev.len()
                        && pm[j]
                    {
                        let v = *z * prev[j].conj();
                        c.ac += Complex64::new(v.re as f64, v.im as f64);
                        c.pairs += 1;
                        c.variance += z.norm_sqr() as f64 * prev[j].norm_sqr() as f64;
                    }
                }
            }
            previous = Some((t[i], fr.header.gate_front_us, y.to_vec(), mask));
            used += 1;
        }
        println!("file {file_index}: {used} accepted pulses");
    }
    let mut result = vec![Cell::default(); 400];
    let mut rejected = vec![0u64; 400];
    let mut sec_reject = vec![0u8; 60 * 400];
    for s in 0..60 {
        for b in 0..400 {
            let c = &grid[s * 400 + b];
            // Conservative coherent-scatter candidate mask, applied to whole blocks.
            if coherent_candidate(c) {
                rejected[b] += 1;
                sec_reject[s * 400 + b] = 1;
                continue;
            }
            let r = &mut result[b];
            r.p += c.p;
            r.noise += c.noise;
            r.n += c.n;
            r.ac += c.ac;
            r.pairs += c.pairs;
            r.variance += c.variance;
        }
    }
    if used==0 {return Ok((epoch,last_file,0));}
    let out = hdf5::File::create(format!("{prefix}.h5"))?;
    let mut snr = Vec::new();
    let mut phase = Vec::new();
    let mut freq = Vec::new();
    let mut coherence = Vec::new();
    let mut significance = Vec::new();
    for c in &result {
        snr.push(if c.n > 0 {
            c.p / c.noise - 1.
        } else {
            f64::NAN
        });
        let rho = if c.n > 0 && c.pairs > 0 {
            c.ac.norm() / c.pairs as f64 / (c.p / c.n as f64)
        } else {
            f64::NAN
        };
        coherence.push(rho);
        // Neighbouring range gates are correlated. 800-sample inflation is
        // conservative, not a calibrated confidence interval.
        let sig = if c.variance > 0. {
            c.ac.norm() / (400. * c.variance).sqrt()
        } else {
            0.
        };
        significance.push(sig);
        phase.push(if c.pairs > 0 { c.ac.arg() } else { f64::NAN });
        freq.push(if sig >= 5. {
            pulse_pair_hz(c.ac)
        } else {
            f64::NAN
        });
    }
    let ranges: Vec<_> = (0..400).map(|i| (i as f64 + 0.5) * 2.).collect();
    let background_bins: Vec<_> = (0..400)
        .filter(|b| (120.0..180.).contains(&ranges[*b]) || (550.0..600.).contains(&ranges[*b]))
        .collect();
    let mut background_ac = vec![Complex64::default(); 60];
    let mut background_power = vec![0.; 60];
    for s in 0..60 {
        let mut ac = Complex64::default();
        let mut pairs = 0;
        let mut p = 0.;
        let mut count = 0;
        for b in &background_bins {
            let c = &grid[s * 400 + b];
            if sec_reject[s * 400 + b] == 0 {
                ac += c.ac;
                pairs += c.pairs;
                p += c.p;
                count += c.n;
            }
        }
        if pairs > 0 {
            background_ac[s] = ac / pairs as f64;
        }
        if count > 0 {
            background_power[s] = p / count as f64;
        }
    }
    let mut residual_phase = Vec::new();
    let mut residual_frequency = Vec::new();
    let mut residual_sig = Vec::new();
    let mut excess = Vec::new();
    for b in 0..400 {
        let mut ac = Complex64::default();
        let mut weight = 0.;
        let mut points = Vec::new();
        let mut signal = 0.;
        let mut noise = 0.;
        for s in 0..60 {
            let c = &grid[s * 400 + b];
            if sec_reject[s * 400 + b] == 0 && c.pairs > 0 {
                let value = c.ac / c.pairs as f64 - background_ac[s];
                let w = c.pairs as f64;
                ac += value * w;
                weight += w;
                points.push((value, w));
                signal += c.p - background_power[s] * c.n as f64;
                noise += background_power[s] * c.n as f64;
            }
        }
        let mean = if weight > 0. {
            ac / weight
        } else {
            Complex64::default()
        };
        let variance = points
            .iter()
            .map(|(v, w)| (*w / weight).powi(2) * (*v - mean).norm_sqr())
            .sum::<f64>()
            * points.len() as f64
            / (points.len().max(2) - 1) as f64;
        let sig = if variance > 0. {
            mean.norm() / variance.sqrt()
        } else {
            0.
        };
        residual_phase.push(if weight > 0. { mean.arg() } else { f64::NAN });
        residual_sig.push(sig);
        residual_frequency.push(if sig >= 5. && points.len() >= 20 {
            mean.arg() / (std::f64::consts::TAU * 0.005)
        } else {
            f64::NAN
        });
        excess.push(if noise > 0. { signal / noise } else { f64::NAN });
    }
    for (name, v) in [
        ("slant_range_km", &ranges),
        ("signal_to_noise_signed", &snr),
        ("lag1_phase_rad", &phase),
        ("aliased_frequency_hz_significant_only", &freq),
        ("lag1_coherence", &coherence),
        ("lag1_significance_conservative", &significance),
        ("off_echo_subtracted_phase_rad", &residual_phase),
        (
            "off_echo_subtracted_frequency_hz_significant_only",
            &residual_frequency,
        ),
        ("off_echo_subtracted_block_significance", &residual_sig),
        ("off_echo_subtracted_signal_to_noise_signed", &excess),
    ] {
        out.new_dataset_builder().with_data(v).create(name)?;
    }
    for (name, v) in [
        (
            "power_adc2",
            result
                .iter()
                .map(|r| if r.n > 0 { r.p / r.n as f64 } else { f64::NAN })
                .collect::<Vec<_>>(),
        ),
        (
            "noise_adc2",
            result
                .iter()
                .map(|r| {
                    if r.n > 0 {
                        r.noise / r.n as f64
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        ),
        (
            "lag1_real",
            result
                .iter()
                .map(|r| {
                    if r.pairs > 0 {
                        r.ac.re / r.pairs as f64
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        ),
        (
            "lag1_imag",
            result
                .iter()
                .map(|r| {
                    if r.pairs > 0 {
                        r.ac.im / r.pairs as f64
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        ),
    ] {
        out.new_dataset_builder().with_data(&v).create(name)?;
    }
    out.new_dataset_builder()
        .with_data(&rejected)
        .create("coherent_candidate_blocks_rejected")?;
    out.new_dataset::<u8>()
        .shape([60, 400])
        .create("coherent_candidate_mask")?
        .write_raw(&sec_reject)?;
    out.new_dataset_builder()
        .with_data(&result.iter().map(|r| r.n).collect::<Vec<_>>())
        .create("sample_counts")?;
    out.new_dataset_builder()
        .with_data(&result.iter().map(|r| r.pairs).collect::<Vec<_>>())
        .create("pair_counts")?;
    out.new_dataset_builder()
        .with_data(&[epoch, epoch + 60_000_000_000])
        .create("utc_bounds_ns")?;
    out.new_dataset_builder()
        .with_data(&[az, el])
        .create("tx_azimuth_elevation_deg")?;
    out.new_dataset_builder()
        .with_data(&background_power)
        .create("off_echo_power_by_second")?;
    let late_noise: Vec<_> = late_power
        .iter()
        .zip(&late_count)
        .map(|(p, n)| if *n > 0 { p / *n as f64 } else { f64::NAN })
        .collect();
    out.new_dataset_builder()
        .with_data(&late_noise)
        .create("late_gate_noise_power_by_second")?;
    out.new_dataset_builder()
        .with_data(&late_count)
        .create("late_gate_noise_sample_count")?;
    out.new_dataset_builder()
        .with_data(&[late_delay_min, late_delay_max])
        .create("late_gate_noise_delay_bounds_us")?;
    out.new_dataset_builder().with_data(b"Last 800 valid matched-filter lags per pulse (200 us); arithmetic mean after meteor pulse exclusion and instantaneous 20 dB mask. This is the latest full-pulse reference window available in the recording, not the final 200 us of the physical 5 ms IPP.").create("late_gate_noise_method_utf8")?;
    for (name, v) in [
        (
            "block_power_sum",
            grid.iter().map(|c| c.p).collect::<Vec<_>>(),
        ),
        (
            "block_lag1_real_sum",
            grid.iter().map(|c| c.ac.re).collect(),
        ),
        (
            "block_lag1_imag_sum",
            grid.iter().map(|c| c.ac.im).collect(),
        ),
    ] {
        out.new_dataset::<f64>()
            .shape([60, 400])
            .create(name)?
            .write_raw(&v)?;
    }
    for (name, v) in [
        (
            "block_sample_count",
            grid.iter().map(|c| c.n).collect::<Vec<_>>(),
        ),
        ("block_pair_count", grid.iter().map(|c| c.pairs).collect()),
    ] {
        out.new_dataset::<u64>()
            .shape([60, 400])
            .create(name)?
            .write_raw(&v)?;
    }
    out.new_dataset_builder().with_data(b"Diagnostic background subtraction uses 120-180 and 550-600 km, independently each second. Both power and complex lag-1 background subtracted. Residual significance from scatter of independent 1-second block complex means, exposure-weighted; not calibrated confidence coverage. Off-echo subtraction does not establish instrumental origin or plasma Doppler calibration.").create("background_method_utf8")?;
    out.new_dataset_builder().with_data(b"Q+iI; lag1=y(next)*conj(y(previous)); 5 ms; f=arg(lag1)/(2pi*IPP); monostatic v=-lambda*f/2; no alias branch selected; signed noise-subtracted power; coherence mask is heuristic, not proof of incoherent origin; cross-range covariance significance inflation=800; clocks UTC+8 to UTC; absolute delay uncalibrated").create("conventions_utf8")?;
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 170.;
    fig.height_mm = 145.;
    for (row, (values, label, bounds)) in [
        (&excess, "Excess / background", [-0.03, 0.03]),
        (&coherence, "Lag-1 coherence", [0., 0.2]),
        (&residual_phase, "Residual phase (rad)", [-3.2, 3.2]),
    ]
    .into_iter()
    .enumerate()
    {
        let max = if row == 0 {
            ranges
                .iter()
                .zip(&excess)
                .filter(|(r, x)| **r >= 80. && **r <= 140. && x.is_finite())
                .map(|(_, x)| *x)
                .fold(0.03, f64::max)
                * 1.1
        } else {
            bounds[1]
        };
        let mut p = Panel::new(
            Rect::new(27., 7. + row as f64 * 43., 135., 32.),
            Axes::new([80., 140.], [bounds[0], max], "Slant range (km)", label),
        );
        p.push(Layer::Line(Line::new(
            ranges
                .iter()
                .zip(values)
                .filter(|(r, _)| **r >= 80. && **r <= 140.)
                .map(|(r, v)| [*r, *v])
                .collect(),
            "#0072B2",
        )));
        fig.panels.push(p);
    }
    fig.save(format!("{prefix}.png"))?;
    println!("used={used}, meteor-excluded={excluded}, coherent candidate blocks={}, significant phase bins={}",rejected.iter().sum::<u64>(),freq.iter().filter(|v|v.is_finite()).count());
    println!(
        "off-echo-subtracted bins above diagnostic threshold: {}",
        residual_frequency.iter().filter(|f| f.is_finite()).count()
    );
    Ok((epoch,last_file,used))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn phase_sign_and_aliases() {
        for (truth, aliased) in [
            (25., 25.),
            (-25., -25.),
            (225., 25.),
            (-175., 25.),
            (125., -75.),
        ] {
            let a = Complex64::from_polar(1., 0.7);
            let b = Complex64::from_polar(1., 0.7 + std::f64::consts::TAU * truth * 0.005);
            assert!((pulse_pair_hz(b * a.conj()) - aliased).abs() < 1e-10);
        }
    }
    #[test]
    fn coherent_mask_rejects_strong_correlated_echo_not_zero_lag_only_power() {
        let mut c = Cell {
            p: 200.,
            noise: 100.,
            n: 100,
            ac: Complex64::new(100., 0.),
            pairs: 100,
            variance: 0.,
        };
        assert!(coherent_candidate(&c));
        c.ac = Complex64::default();
        assert!(!coherent_candidate(&c));
    }
}
