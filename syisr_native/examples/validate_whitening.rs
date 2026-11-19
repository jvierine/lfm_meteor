#[path = "../src/filter.rs"]
#[allow(dead_code)]
mod filter;
#[path = "../src/raw.rs"]
#[allow(dead_code)]
mod raw;
#[path = "../src/rfi.rs"]
mod rfi;
use anyhow::{ensure, Result};
use num_complex::{Complex32, Complex64};
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "usage: validate_whitening PRODUCTS OUTPUT.h5");
    let index = hdf5::File::with_options()
        .with_fapl(|p| p.core_filebacked(false))
        .open(Path::new(&a[1]).join("work/pulses_000000.h5"))?;
    let path = String::from_utf8(index.dataset("source_path_utf8")?.read_raw::<u8>()?)?;
    let offsets = index
        .dataset("pulses/frame_byte_offset")?
        .read_raw::<u64>()?;
    let mut reader = raw::Reader::open(Path::new(&path), 8, 8)?;
    let mut frames = Vec::new();
    let mut spectra = Vec::new();
    let mut offset_estimates = Vec::new();
    for offset in offsets.iter().skip(1).take(512) {
        reader.seek(*offset)?;
        let fr = reader.next()?.unwrap();
        let mut clean = fr.iq.clone();
        offset_estimates.push(rfi::remove_offsets(&mut clean));
        spectra.push(rfi::periodogram(&clean));
        frames.push((fr.header.utc_ns, fr.iq, clean));
    }
    let psd = rfi::robust_psd(&spectra[..128]);
    let taps = rfi::whitening_taps(&psd);
    let n = frames[0].1.len();
    let mut original = filter::Filter::new(n, true);
    let mut white = filter::Filter::whitened(n, true, &taps);
    let mut p = [0.; 2];
    let mut ac = [Complex64::default(); 2];
    let mut counts = 0u64;
    let mut prev: Option<(i64, Vec<Complex32>, Vec<Complex32>)> = None;
    let mut cleaned_spectrum = vec![0.; rfi::PSD_N];
    for (i, (utc, z, clean)) in frames.iter().enumerate() {
        let y = original.apply(z).to_vec();
        let yc = white.apply(clean).to_vec();
        if let Some((t, a, b)) = &prev {
            if utc - t == 5_000_000 {
                for j in y.len() - 1600..y.len() - 800 {
                    for (q, (v, w)) in [(y[j], a[j]), (yc[j], b[j])].into_iter().enumerate() {
                        p[q] += v.norm_sqr() as f64;
                        let v = v * w.conj();
                        ac[q] += Complex64::new(v.re as f64, v.im as f64);
                    }
                    counts += 1;
                }
            }
        }
        prev = Some((*utc, y, yc));
        for (s, p) in cleaned_spectrum.iter_mut().zip(&spectra[i]) {
            *s += *p as f64 / frames.len() as f64;
        }
    }
    let before = ac[0].norm() / p[0];
    let after = ac[1].norm() / p[1];
    let out = hdf5::File::create(&a[2])?;
    out.new_dataset_builder()
        .with_data(&psd)
        .create("robust_training_psd")?;
    out.new_dataset_builder()
        .with_data(&cleaned_spectrum)
        .create("offset_removed_mean_psd")?;
    out.new_dataset_builder()
        .with_data(&[before, after])
        .create("off_echo_coherence_before_after")?;
    out.new_dataset_builder()
        .with_data(&[ac[0].arg(), ac[1].arg()])
        .create("off_echo_phase_before_after_rad")?;
    out.new_dataset_builder()
        .with_data(&taps.iter().map(|v| v.re).collect::<Vec<_>>())
        .create("whitener_taps_real")?;
    out.new_dataset_builder()
        .with_data(&taps.iter().map(|v| v.im).collect::<Vec<_>>())
        .create("whitener_taps_imag")?;
    out.new_dataset_builder()
        .with_data(&[white.reference_energy])
        .create("whitened_reference_energy")?;
    out.new_dataset_builder().with_data(b"512 raw pulses; training first 128, evaluation all 512; tail-based robust complex DC and Nyquist offset removal; 257-tap Hann-whitening FIR applied to both voltage and reference through its squared frequency response; extra zero padding prevents circular wrap. Coherence evaluated on late full-pulse lags with 200 us edge guard; diagnostic, not independent validation of plasma velocity.").create("methods_utf8")?;
    println!("{counts} off-echo pulse/gate pairs; coherence {:.6} -> {:.6}; phase {:.4} -> {:.4} rad; reference energy {:.3}",before,after,ac[0].arg(),ac[1].arg(),white.reference_energy);
    ensure!(
        after < before * 0.3,
        "offset correction did not sufficiently suppress common correlation"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn predicted_noise_power_matches_linear_filter_impulse_energy() {
        let mut taps = rfi::whitening_taps(&vec![1.; rfi::PSD_N]);
        rfi::notch_offsets(&mut taps);
        let mut f = filter::Filter::whitened(5000, true, &taps);
        let expected = f.noise_power(&vec![1.; rfi::PSD_N]);
        let mut impulse = vec![Complex32::default(); 5000];
        impulse[2500] = Complex32::new(1., 0.);
        let energy = f
            .apply(&impulse)
            .iter()
            .map(|v| v.norm_sqr() as f64)
            .sum::<f64>();
        assert!((energy / expected - 1.).abs() < 1e-5);
    }
    #[test]
    fn whitened_echo_keeps_delay_and_known_gain() {
        let mut psd = vec![1.; rfi::PSD_N];
        psd[128] = 100.;
        let taps = rfi::whitening_taps(&psd);
        let mut z = vec![Complex32::default(); 5000];
        z[1500..2300].copy_from_slice(&filter::chirp(true));
        let mut f = filter::Filter::whitened(z.len(), true, &taps);
        let energy = f.reference_energy;
        let y = f.apply(&z);
        let k = (0..y.len())
            .max_by(|a, b| y[*a].norm_sqr().total_cmp(&y[*b].norm_sqr()))
            .unwrap();
        assert_eq!(k, 1500);
        assert!((y[k].norm() as f64 / energy - 1.).abs() < 1e-5);
        let side = y
            .iter()
            .enumerate()
            .filter(|(i, _)| i.abs_diff(k) > 2)
            .map(|(_, v)| v.norm())
            .fold(0., f32::max);
        assert!(side / y[k].norm() < 0.08);
    }
    #[test]
    fn finite_whitening_equals_explicit_linear_convolution() {
        let taps = vec![
            Complex32::new(0.2, 0.1),
            Complex32::new(1., 0.),
            Complex32::new(0.2, -0.1),
        ];
        let z: Vec<_> = (0..1500)
            .map(|i| Complex32::new((i as f32 * 0.31).sin(), (i as f32 * 0.17).cos()))
            .collect();
        let convolve = |x: &[Complex32]| {
            let mut y = vec![Complex32::default(); x.len() + taps.len() - 1];
            for (i, v) in x.iter().enumerate() {
                for (j, w) in taps.iter().enumerate() {
                    y[i + j] += v * w;
                }
            }
            y
        };
        let wz = convolve(&z);
        let ws = convolve(&filter::chirp(true));
        let mut f = filter::Filter::whitened(z.len(), true, &taps);
        let y = f.apply(&z);
        for k in [0, 100, 700] {
            let direct: Complex32 = wz[k..k + ws.len()]
                .iter()
                .zip(&ws)
                .map(|(z, s)| z * s.conj())
                .sum();
            assert!((direct - y[k]).norm() < 0.002);
        }
    }
}
