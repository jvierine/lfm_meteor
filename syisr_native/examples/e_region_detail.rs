#[path = "../src/filter.rs"]
#[allow(dead_code)]
mod filter;
#[path = "../src/raw.rs"]
#[allow(dead_code)]
mod raw;
#[path = "../src/rfi.rs"]
#[allow(dead_code)]
mod rfi;
use anyhow::{ensure, Result};
use num_complex::Complex32;
use rustplotlib::*;
use std::path::Path;
fn valid_times(t: &[i64]) -> Vec<bool> {
    let mut tails: Vec<usize> = Vec::new();
    let mut prev = vec![usize::MAX; t.len()];
    for (i, v) in t.iter().enumerate() {
        let k = tails.partition_point(|j| t[*j] < *v);
        if k > 0 {
            prev[i] = tails[k - 1];
        }
        if k == tails.len() {
            tails.push(i);
        } else {
            tails[k] = i;
        }
    }
    let mut valid = vec![false; t.len()];
    if let Some(j) = tails.last() {
        let mut i = *j;
        loop {
            valid[i] = true;
            if prev[i] == usize::MAX {
                break;
            }
            i = prev[i];
        }
    }
    valid
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 4,
        "usage: e_region_detail METEOR_PRODUCTS IONOSPHERE_PRODUCTS OUTPUT_PREFIX"
    );
    let rawroot = Path::new(&a[1]);
    let root = Path::new(&a[2]);
    let cal = hdf5::File::open(root.join("rti_noise_calibrated.h5"))?;
    let times = cal.dataset("window_start_utc_ns")?.read_raw::<i64>()?;
    let noise = cal.dataset("calibrated_noise_adc2")?.read_raw::<f64>()?;
    let cat = hdf5::File::open(rawroot.join("catalogue.h5"))?;
    let starts = cat.dataset("event_start_utc_ns")?.read_raw::<i64>()?;
    let stops = cat.dataset("event_stop_utc_ns")?.read_raw::<i64>()?;
    let epoch = chrono::NaiveDate::from_ymd_opt(2026, 11, 17)
        .unwrap()
        .and_hms_opt(13, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp()
        * 1_000_000_000;
    let end = epoch + 600_000_000_000;
    let nr = 534;
    let nt = 60;
    let dr = 0.15;
    let mut p = vec![0f64; nt * nr];
    let mut n = vec![0u64; nt * nr];
    let mut bg = vec![0f64; nt * nr];
    let mut active_minute = usize::MAX;
    let mut active_n = 0;
    let mut compressor: Option<filter::Filter> = None;
    let mut mask = Vec::new();
    let mut taps = Vec::new();
    let mut el = 75.;
    let mut used = 0;
    let mut last_time = 0i64;
    for file in 97..854 {
        let index = hdf5::File::with_options()
            .with_fapl(|p| p.core_filebacked(false))
            .open(rawroot.join(format!("work/pulses_{file:06}.h5")))?;
        let t = index.dataset("pulses/utc_ns")?.read_raw::<i64>()?;
        let flags = valid_times(&t);
        let offsets = index
            .dataset("pulses/frame_byte_offset")?
            .read_raw::<u64>()?;
        let first = *t.iter().zip(&flags).find(|(_, v)| **v).unwrap().0;
        if first >= end {
            break;
        }
        let path = String::from_utf8(index.dataset("source_path_utf8")?.read_raw::<u8>()?)?;
        let mut reader = raw::Reader::open(Path::new(&path), 8, 8)?;
        for i in 0..t.len() {
            let utc = t[i];
            if !flags[i] || utc < epoch || utc >= end || utc <= last_time {
                continue;
            }
            if starts
                .iter()
                .zip(&stops)
                .any(|(s, e)| utc >= s - 100_000_000 && utc <= e + 100_000_000)
            {
                continue;
            }
            let minute = times.partition_point(|t| *t <= utc).saturating_sub(1);
            let second = ((utc - times[minute]) / 1_000_000_000) as usize;
            if second >= 60 || !noise[minute * 60 + second].is_finite() {
                continue;
            }
            if minute != active_minute {
                let h = hdf5::File::with_options()
                    .with_fapl(|p| p.core_filebacked(false))
                    .open(root.join(format!("minutes/minute_{minute:04}.h5")))?;
                let re = h.dataset("whitening_fir_real")?.read_raw::<f32>()?;
                let im = h.dataset("whitening_fir_imag")?.read_raw::<f32>()?;
                taps = re
                    .iter()
                    .zip(im)
                    .map(|(re, im)| Complex32::new(*re, im))
                    .collect();
                mask = h.dataset("coherent_candidate_mask")?.read_raw::<u8>()?;
                active_minute = minute;
                active_n = 0;
            }
            reader.seek(offsets[i])?;
            let mut fr = reader.next()?.unwrap();
            ensure!(fr.header.utc_ns == utc, "timestamp mismatch");
            el = fr.header.el_deg;
            rfi::remove_offsets(&mut fr.iq);
            if active_n != fr.iq.len() {
                compressor = Some(filter::Filter::whitened(fr.iq.len(), true, &taps));
                active_n = fr.iq.len();
            }
            let y = compressor.as_mut().unwrap().apply(&fr.iq);
            let time = ((utc - epoch) / 10_000_000_000) as usize;
            let q = noise[minute * 60 + second];
            for (j, z) in y.iter().enumerate().skip(256).take(y.len() - 512) {
                let range = filter::C / 2e9 * (fr.header.gate_front_us + j as f64 / 4.);
                if !(80.0..160.1).contains(&range) {
                    continue;
                }
                let coarse = (range / 2.).floor() as usize;
                if mask[second * 400 + coarse] != 0 || z.norm_sqr() as f64 > 100. * q {
                    continue;
                }
                let b = ((range - 80.) / dr).floor() as usize;
                if b >= nr {
                    continue;
                }
                let k = time * nr + b;
                p[k] += z.norm_sqr() as f64;
                bg[k] += q;
                n[k] += 1;
            }
            used += 1;
            last_time = utc;
        }
        println!("file {file}: {used} accepted pulses");
    }
    let power: Vec<_> = p
        .iter()
        .zip(&bg)
        .map(|(p, b)| if *b > 0. { p / b } else { f64::NAN })
        .collect();
    let out = hdf5::File::create(format!("{}.h5", a[3]))?;
    out.new_dataset::<f64>()
        .shape([nt, nr])
        .create("power_over_noise")?
        .write_raw(&power)?;
    out.new_dataset::<u64>()
        .shape([nt, nr])
        .create("sample_counts")?
        .write_raw(&n)?;
    let ranges: Vec<_> = (0..nr).map(|b| 80. + (b as f64 + 0.5) * dr).collect();
    out.new_dataset_builder()
        .with_data(&ranges)
        .create("slant_range_km")?;
    out.new_dataset_builder()
        .with_data(
            &(0..nt)
                .map(|t| epoch + t as i64 * 10_000_000_000)
                .collect::<Vec<_>>(),
        )
        .create("window_start_utc_ns")?;
    out.new_dataset_builder()
        .with_data(&[el])
        .create("elevation_deg")?;
    out.new_dataset_builder().with_data(b"Raw-voltage replay, 150 m slant-range bins and 10-second averages, 13:00-13:10 UTC. Same per-minute finite-FIR whitener and global DC/Nyquist removal as full reduction. Noise from held-out tail calibration. Original meteor intervals and one-second/2-km scatter masks retained; additional per-gate 20 dB instantaneous threshold. No interpolation. Range bins are not independent resolution elements.").create("methods_utf8")?;
    let altitude =
        |r: f64| (6371f64.powi(2) + r * r + 2. * 6371. * r * el.to_radians().sin()).sqrt() - 6371.;
    let lo = ranges
        .iter()
        .position(|r| altitude(*r + dr / 2.) >= 80.)
        .unwrap();
    let hi = ranges
        .iter()
        .position(|r| altitude(*r - dr / 2.) >= 150.)
        .unwrap();
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 170.;
    fig.height_mm = 100.;
    let scale = fig.scale(Scale::new(
        [0.97, 1.06],
        Normalization::Linear,
        ColorMap::storm(),
    )?);
    let mut panel = Panel::new(
        Rect::new(23., 8., 116., 72.),
        Axes::new(
            [13., 13. + 1. / 6.],
            [80., 150.],
            "UTC hour (2026-11-17)",
            "Altitude (km)",
        ),
    );
    let field = (lo..hi)
        .flat_map(|b| {
            let data = &power;
            (0..nt).map(move |t| data[t * nr + b])
        })
        .collect();
    panel.push(Layer::Pcolormesh(Field::new(
        (0..=nt).map(|t| 13. + t as f64 / 360.).collect(),
        (lo..=hi).map(|b| altitude(ranges[b] - dr / 2.)).collect(),
        field,
        scale,
    )?));
    fig.panels.push(panel);
    fig.colorbars.push(Colorbar::new(
        Rect::new(142., 8., 3., 72.),
        scale,
        "Power / noise",
        vec![0.98, 1., 1.02, 1.04, 1.06],
    ));
    fig.save(format!("{}.png", a[3]))?;
    println!("COMPLETE: {used} pulses");
    Ok(())
}
