// Replay original Q+iI voltage; no access to the original recording is needed.
#[path = "../src/filter.rs"]
#[allow(dead_code)]
mod filter;
use anyhow::{ensure, Result};
use num_complex::Complex32;
use rustplotlib::*;

fn refine(z: &[Complex32], coarse: f64, fd: f64, down: bool) -> (f64, f64) {
    let mut best = (0., coarse);
    for step in -32..=32 {
        let lag = coarse + step as f64 / 32.;
        let start = lag.ceil().max(0.) as usize;
        let stop = ((lag + 800.).ceil() as usize).min(z.len());
        if stop - start != 800 {
            continue;
        }
        let mut sum = num_complex::Complex64::new(0., 0.);
        for (j, value) in z.iter().enumerate().take(stop).skip(start) {
            let t = (j as f64 - lag) / filter::FS;
            let phase = std::f64::consts::TAU
                * ((if down { 1. } else { -1. }) * (2e6 * t - 1e10 * t * t)
                    + fd * j as f64 / filter::FS);
            sum += num_complex::Complex64::new(value.re as f64, value.im as f64)
                * num_complex::Complex64::from_polar(1., -phase);
        }
        if sum.norm_sqr() > best.0 {
            best = (sum.norm_sqr(), lag);
        }
    }
    best
}

fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "usage: replay_cut EVENT.h5 OUTPUT_PREFIX");
    let h = hdf5::File::open(&a[1])?;
    let read = |n: &str| h.dataset(n)?.read_raw::<f64>();
    let words = h.dataset("raw/iq_words_iq")?.read_raw::<i32>()?;
    let offsets = h.dataset("raw/pulse_offsets")?.read_raw::<u64>()?;
    let first = h.dataset("raw/first_sample")?.read_raw::<u64>()?;
    let delay = read("raw/first_delay_us")?;
    let times = h.dataset("pulses/utc_ns")?.read_raw::<i64>()?;
    let epoch = h.dataset("track_epoch_utc_ns")?.read_raw::<i64>()?[0];
    let rate = read("track_range_rate_km_s")?[0];
    let km = read("range_km_per_delay_us")?[0];
    let carrier = read("instrument/carrier_frequency_hz")?[0];
    let track = read("predicted_track_range_km")?;
    let config: toml::Value = toml::from_str(&String::from_utf8(
        h.dataset("config_toml_utf8")?.read_raw::<u8>()?,
    )?)?;
    let down = config["chirp_direction"].as_str() == Some("down");
    // km/us converts the displayed coordinate to total delay, for either geometry.
    let fd = -(rate / km) * 1e-6 * carrier;
    let cached_re = h.dataset("matched_filter/real")?.read_raw::<f32>()?;
    let cached_im = h.dataset("matched_filter/imag")?.read_raw::<f32>()?;
    let cached_offsets = h
        .dataset("matched_filter/pulse_offsets")?
        .read_raw::<u64>()?;
    let cached_first = h
        .dataset("matched_filter/first_lag_sample")?
        .read_raw::<u64>()?;
    let mut original = Vec::new();
    let mut corrected = Vec::new();
    let mut ranges = Vec::new();
    let mut refined_original = Vec::new();
    let mut refined_corrected = Vec::new();
    let mut strongest_voltage = Vec::new();
    let mut strongest_index = 0;
    let mut strongest_power = 0.;
    let mut max_relative_error = 0f64;
    for i in 0..times.len() {
        let z: Vec<_> = words[2 * offsets[i] as usize..2 * offsets[i + 1] as usize]
            .chunks_exact(2)
            .map(|w| Complex32::new(w[1] as f32, w[0] as f32))
            .collect();
        let mut f = filter::Filter::new(z.len(), down);
        let y = f.apply(&z).to_vec();
        let mut error = 0.;
        let mut norm = 0.;
        for j in cached_offsets[i] as usize..cached_offsets[i + 1] as usize {
            let k = cached_first[i] as usize + j - cached_offsets[i] as usize - first[i] as usize;
            let expected = Complex32::new(cached_re[j], cached_im[j]);
            error += (y[k] - expected).norm_sqr() as f64;
            norm += expected.norm_sqr() as f64;
        }
        max_relative_error = max_relative_error.max((error / norm).sqrt());
        let shifted: Vec<_> = z
            .iter()
            .enumerate()
            .map(|(j, z)| {
                *z * Complex32::from_polar(
                    1.,
                    (-std::f64::consts::TAU * fd * j as f64 / filter::FS) as f32,
                )
            })
            .collect();
        let yc = f.apply(&shifted);
        let peak = |v: &[Complex32]| -> (f64, f64) {
            let k = (0..v.len())
                .filter(|j| (km * (delay[i] + *j as f64 / 4.) - track[i]).abs() < 2.)
                .max_by(|x, y| v[*x].norm_sqr().total_cmp(&v[*y].norm_sqr()))
                .unwrap();
            (v[k].norm_sqr() as f64, km * (delay[i] + k as f64 / 4.))
        };
        let (p, r) = peak(&y);
        let (pc, rc) = peak(yc);
        refined_original.push(refine(&z, (r / km - delay[i]) * 4., 0., down).0);
        refined_corrected.push(refine(&z, (rc / km - delay[i]) * 4., fd, down).0);
        if p > strongest_power {
            strongest_power = p;
            strongest_voltage = z;
            strongest_index = i;
        }
        original.push(p);
        corrected.push(pc);
        ranges.push([r, rc]);
    }
    ensure!(
        max_relative_error < 1e-4,
        "cached-filter replay mismatch: {max_relative_error}"
    );
    let out = hdf5::File::create(format!("{}.h5", a[2]))?;
    for (n, v) in [
        ("zero_doppler_peak_power", &original),
        ("migration_doppler_peak_power", &corrected),
        ("refined_zero_doppler_peak_power", &refined_original),
        ("refined_migration_doppler_peak_power", &refined_corrected),
    ] {
        out.new_dataset_builder().with_data(v).create(n)?;
    }
    out.new_dataset_builder()
        .with_data(&[fd])
        .create("predicted_doppler_hz")?;
    out.new_dataset_builder()
        .with_data(&[max_relative_error])
        .create("replay_relative_rms_error")?;
    out.new_dataset_builder()
        .with_data(&times)
        .create("utc_ns")?;
    out.new_dataset_builder()
        .with_data(&ranges.iter().map(|r| r[0]).collect::<Vec<_>>())
        .create("zero_doppler_peak_range_km")?;
    out.new_dataset_builder()
        .with_data(&ranges.iter().map(|r| r[1]).collect::<Vec<_>>())
        .create("migration_doppler_peak_range_km")?;
    // Full delay/Doppler bank for one pulse. This exposes the LFM ambiguity;
    // the bank maximum alone is not an independent precise Doppler measurement.
    let i = strongest_index;
    let mut bank = Vec::new();
    let mut lag_ranges = Vec::new();
    let mut frequencies = Vec::new();
    let mut bf = filter::Filter::new(strongest_voltage.len(), down);
    for row in 0..=100 {
        let frequency = -250_000. + row as f64 * 5000.;
        frequencies.push(frequency);
        let shifted: Vec<_> = strongest_voltage
            .iter()
            .enumerate()
            .map(|(j, z)| {
                *z * Complex32::from_polar(
                    1.,
                    (-std::f64::consts::TAU * frequency * j as f64 / filter::FS) as f32,
                )
            })
            .collect();
        let response = bf.apply(&shifted);
        for (j, v) in response.iter().enumerate() {
            let r = km * (delay[i] + j as f64 / 4.);
            if (r - track[i]).abs() < 2. {
                if row == 0 {
                    lag_ranges.push(r);
                }
                bank.push(v.norm_sqr() as f64);
            }
        }
    }
    out.new_dataset_builder()
        .with_data(&frequencies)
        .create("bank_doppler_hz")?;
    out.new_dataset_builder()
        .with_data(&lag_ranges)
        .create("bank_range_km")?;
    out.new_dataset::<f64>()
        .shape([frequencies.len(), lag_ranges.len()])
        .create("bank_power")?
        .write_raw(&bank)?;
    out.new_dataset_builder()
        .with_data(&[i as u64])
        .create("bank_pulse_index")?;
    let mut bankfig = Figure::new(IntendedUse::ArticleWide);
    bankfig.width_mm = 160.;
    bankfig.height_mm = 100.;
    let peak_power = bank.iter().copied().fold(0., f64::max);
    let scale = bankfig.scale(Scale::new(
        [-25., 0.],
        Normalization::Linear,
        ColorMap::sequential(),
    )?);
    let dr = km / 4.;
    let mut bp = Panel::new(
        Rect::new(24., 8., 107., 72.),
        Axes::new(
            [
                lag_ranges[0] - dr / 2.,
                *lag_ranges.last().unwrap() + dr / 2.,
            ],
            [-252.5, 252.5],
            "Trial echo range (km)",
            "Trial Doppler (kHz)",
        ),
    );
    bp.push(Layer::Pcolormesh(Field::new(
        (0..=lag_ranges.len())
            .map(|j| lag_ranges[0] + (j as f64 - 0.5) * dr)
            .collect(),
        (0..=101).map(|j| -252.5 + j as f64 * 5.).collect(),
        bank.iter()
            .map(|p| 10. * (p / peak_power).max(1e-10).log10())
            .collect(),
        scale,
    )?));
    let mut guide = Line::new(
        vec![
            [lag_ranges[0], fd / 1000.],
            [*lag_ranges.last().unwrap(), fd / 1000.],
        ],
        "#D55E00",
    );
    guide.width_pt = 1.;
    bp.push(Layer::Line(guide));
    bankfig.panels.push(bp);
    bankfig.colorbars.push(Colorbar::new(
        Rect::new(134., 8., 3., 72.),
        scale,
        "Power (dB)",
        vec![-20., -10., 0.],
    ));
    bankfig.save(format!("{}_bank.png", a[2]))?;
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 160.;
    fig.height_mm = 85.;
    let ts: Vec<_> = times.iter().map(|t| (*t - epoch) as f64 * 1e-6).collect();
    let normalization = original.iter().copied().fold(0., f64::max);
    let mut panel = Panel::new(
        Rect::new(22., 8., 130., 58.),
        Axes::new(
            [ts[0], *ts.last().unwrap()],
            [-30., 3.],
            "Time (ms)",
            "Peak power (dB, relative)",
        ),
    );
    for (values, color) in [
        (&refined_original, "#0072B2"),
        (&refined_corrected, "#D55E00"),
    ] {
        panel.push(Layer::Line(Line::new(
            ts.iter()
                .zip(values)
                .map(|(t, p)| [*t, 10. * (p / normalization).log10()])
                .collect(),
            color,
        )));
    }
    fig.panels.push(panel);
    fig.save(format!("{}.png", a[2]))?;
    let k = (0..original.len())
        .max_by(|i, j| original[*i].total_cmp(&original[*j]))
        .unwrap();
    println!("range rate={rate:.3} km/s; Doppler={fd:.1} Hz; replay relative RMS={max_relative_error:.3e}; strongest-pulse gain={:.3} dB; range shift={:.1} m",10.*(corrected[k]/original[k]).log10(),1000.*(ranges[k][1]-ranges[k][0]));
    println!(
        "fractional-delay refined strongest-pulse gain={:.3} dB",
        10. * (refined_corrected[k] / refined_original[k]).log10()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fractional_delay_search_recovers_known_echo() {
        for down in [true, false] {
            for fd in [-125000., 125000.] {
                let lag = 700.375;
                let z: Vec<_> = (0..2000)
                    .map(|j| {
                        let t = (j as f64 - lag) / filter::FS;
                        if !(0.0..200e-6).contains(&t) {
                            return Complex32::default();
                        }
                        let phase = std::f64::consts::TAU
                            * ((if down { 1. } else { -1. }) * (2e6 * t - 1e10 * t * t)
                                + fd * j as f64 / filter::FS);
                        Complex32::from_polar(1., phase as f32)
                    })
                    .collect();
                let (power, recovered) = refine(&z, 700., fd, down);
                assert!((recovered - lag).abs() < 1e-9);
                assert!((power / 640000. - 1.).abs() < 1e-6);
            }
        }
    }
}
