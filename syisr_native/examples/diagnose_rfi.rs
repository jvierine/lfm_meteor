#[path = "../src/raw.rs"]
#[allow(dead_code)]
mod raw;
use anyhow::{ensure, Result};
use num_complex::{Complex32, Complex64};
use rustfft::FftPlanner;
use rustplotlib::*;
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "usage: diagnose_rfi PRODUCTS OUTPUT_PREFIX");
    let index = hdf5::File::open(Path::new(&a[1]).join("work/pulses_000000.h5"))?;
    let path = String::from_utf8(index.dataset("source_path_utf8")?.read_raw::<u8>()?)?;
    let offsets = index
        .dataset("pulses/frame_byte_offset")?
        .read_raw::<u64>()?;
    let mut reader = raw::Reader::open(Path::new(&path), 8, 8)?;
    let n = 4096;
    let mut plan = FftPlanner::new();
    let fft = plan.plan_fft_forward(n);
    let window: Vec<_> = (0..n)
        .map(|j| 0.5 - 0.5 * (std::f32::consts::TAU * j as f32 / (n - 1) as f32).cos())
        .collect();
    let energy = window.iter().map(|w| w * w).sum::<f32>() as f64;
    let mut spectrum = vec![0.; n];
    let mut mean_power = 0.;
    let mut dc_power = 0.;
    let mut used = 0;
    let mut m = Complex64::default();
    let mut adjacent = Complex64::default();
    let mut prev: Option<(i64, Complex64)> = None;
    for offset in offsets.iter().skip(1).take(512) {
        reader.seek(*offset)?;
        let fr = reader.next()?.unwrap();
        let z = &fr.iq[fr.iq.len() - n..];
        let mean = z
            .iter()
            .map(|v| Complex64::new(v.re as f64, v.im as f64))
            .sum::<Complex64>()
            / n as f64;
        m += mean;
        mean_power += z.iter().map(|v| v.norm_sqr() as f64).sum::<f64>() / n as f64;
        dc_power += mean.norm_sqr();
        if let Some((t, p)) = prev {
            if fr.header.utc_ns - t == 5_000_000 {
                adjacent += mean * p.conj();
            }
        }
        prev = Some((fr.header.utc_ns, mean));
        let mut v: Vec<Complex32> = z.iter().zip(&window).map(|(z, w)| *z * *w).collect();
        fft.process(&mut v);
        for (j, z) in v.iter().enumerate() {
            spectrum[j] += z.norm_sqr() as f64 / energy;
        }
        used += 1;
    }
    let freq: Vec<_> = (0..n)
        .map(|j| (j as f64 - n as f64 / 2.) * 4e6 / n as f64)
        .collect();
    let power: Vec<_> = (0..n)
        .map(|j| spectrum[(j + n / 2) % n] / used as f64)
        .collect();
    let mut sorted = power.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[n / 2];
    let peak = power
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    let out = hdf5::File::create(format!("{}.h5", a[2]))?;
    out.new_dataset_builder()
        .with_data(&freq)
        .create("frequency_hz")?;
    out.new_dataset_builder()
        .with_data(&power)
        .create("mean_spectral_power_adc2")?;
    out.new_dataset_builder()
        .with_data(&[dc_power / mean_power])
        .create("mean_dc_power_fraction")?;
    out.new_dataset_builder()
        .with_data(&[m.re / used as f64, m.im / used as f64])
        .create("average_complex_dc")?;
    out.new_dataset_builder()
        .with_data(&[
            adjacent.re / (used - 1) as f64,
            adjacent.im / (used - 1) as f64,
        ])
        .create("adjacent_dc_product")?;
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 160.;
    f.height_mm = 85.;
    let mut p = Panel::new(
        Rect::new(24., 8., 128., 58.),
        Axes::new(
            [-2., 2.],
            [-10., 40.],
            "Baseband frequency (MHz)",
            "Power / median (dB)",
        ),
    );
    p.push(Layer::Line(Line::new(
        freq.iter()
            .zip(&power)
            .map(|(x, y)| [*x / 1e6, 10. * (y / median).log10()])
            .collect(),
        "#0072B2",
    )));
    f.panels.push(p);
    f.save(format!("{}.png", a[2]))?;
    println!("{used} pulses; strongest line {:.1} Hz, {:.2} dB above band median; mean-DC power fraction {:.5}; adjacent DC phase {:.5} rad",freq[peak.0],10.*(peak.1/median).log10(),dc_power/mean_power,adjacent.arg());
    Ok(())
}
