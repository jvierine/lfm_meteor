use num_complex::Complex32;
use rustfft::FftPlanner;
pub const PSD_N: usize = 1024;
pub fn median(v: &mut [f32]) -> f32 {
    let n = v.len();
    v.select_nth_unstable_by(n / 2, f32::total_cmp);
    v[n / 2]
}
/// Robust complex offsets across blocks of the raw gate, resisting sparse echoes.
/// Alternating-sample component is the Nyquist line, not physical Doppler.
pub fn remove_offsets(z: &mut [Complex32]) -> [Complex32; 2] {
    let start = 0;
    let tail = &z[start..];
    let mut dc = Vec::new();
    let mut ny = Vec::new();
    for (block, v) in tail.chunks_exact(64).enumerate() {
        let d = v.iter().copied().sum::<Complex32>() / 64.;
        let a = v
            .iter()
            .enumerate()
            .map(|(j, v)| {
                *v * if (start + block * 64 + j) % 2 == 0 {
                    1.
                } else {
                    -1.
                }
            })
            .sum::<Complex32>()
            / 64.;
        dc.push(d);
        ny.push(a);
    }
    let med = |v: &[Complex32]| {
        let mut re: Vec<_> = v.iter().map(|v| v.re).collect();
        let mut im: Vec<_> = v.iter().map(|v| v.im).collect();
        Complex32::new(median(&mut re), median(&mut im))
    };
    let d = med(&dc);
    let a = med(&ny);
    for (j, z) in z.iter_mut().enumerate() {
        *z -= d + a * if j % 2 == 0 { 1. } else { -1. };
    }
    [d, a]
}
pub fn periodogram(z: &[Complex32]) -> Vec<f32> {
    let mut v: Vec<_> = z[z.len() - PSD_N..]
        .iter()
        .enumerate()
        .map(|(j, z)| {
            *z * (0.5 - 0.5 * (std::f32::consts::TAU * j as f32 / (PSD_N - 1) as f32).cos())
        })
        .collect();
    let mut plan = FftPlanner::new();
    plan.plan_fft_forward(PSD_N).process(&mut v);
    v.iter()
        .map(|v| v.norm_sqr() / (3. * PSD_N as f32 / 8.))
        .collect()
}
pub fn robust_psd(spectra: &[Vec<f32>]) -> Vec<f32> {
    (0..PSD_N)
        .map(|j| {
            let mut values: Vec<_> = spectra.iter().map(|p| p[j]).collect();
            median(&mut values) / std::f32::consts::LN_2
        })
        .collect()
}
/// Finite Hann-tapered impulse response: no brick-wall notches or circular filtering.
pub fn whitening_taps(psd: &[f32]) -> Vec<Complex32> {
    let mut values = psd.to_vec();
    let level = median(&mut values);
    let mut weights: Vec<_> = (0..PSD_N)
        .map(|j| {
            let local = (-2isize..=2)
                .map(|d| psd[(j as isize + d).rem_euclid(PSD_N as isize) as usize])
                .fold(0f32, f32::max);
            Complex32::new((level / local.max(0.2 * level)).sqrt(), 0.)
        })
        .collect();
    let mut plan = FftPlanner::new();
    plan.plan_fft_inverse(PSD_N).process(&mut weights);
    (0..257)
        .map(|j| {
            let lag = j as isize - 128;
            let hann = 0.5 + 0.5 * (std::f32::consts::PI * lag as f32 / 128.).cos();
            weights[lag.rem_euclid(PSD_N as isize) as usize] * (hann / PSD_N as f32)
        })
        .collect()
}
/// Smooth finite-length DC/Nyquist zeros; no brick-wall spectral steps.
pub fn notch_offsets(taps: &mut [Complex32]) {
    let n = taps.len();
    let mut even = Complex32::default();
    let mut odd = even;
    let mut we = 0.;
    let mut wo = 0.;
    for (j, v) in taps.iter().enumerate() {
        let w = 0.5 - 0.5 * (std::f32::consts::TAU * j as f32 / (n - 1) as f32).cos();
        if j % 2 == 0 {
            even += v;
            we += w;
        } else {
            odd += v;
            wo += w;
        }
    }
    for (j, v) in taps.iter_mut().enumerate() {
        let w = 0.5 - 0.5 * (std::f32::consts::TAU * j as f32 / (n - 1) as f32).cos();
        *v -= if j % 2 == 0 {
            even * (w / we)
        } else {
            odd * (w / wo)
        };
    }
}
pub fn noise_shape(psd: &[f32]) -> Vec<f32> {
    (0..psd.len())
        .map(|j| {
            let mut local: Vec<_> = (-8isize..=8)
                .map(|d| psd[(j as isize + d).rem_euclid(psd.len() as isize) as usize])
                .collect();
            median(&mut local)
        })
        .collect()
}
pub fn spectral_scale(periodogram: &[f32], shape: &[f32]) -> (f32, f32) {
    let mut ratios: Vec<_> = periodogram
        .iter()
        .zip(shape)
        .map(|(p, b)| p / b.max(f32::MIN_POSITIVE))
        .collect();
    ratios.sort_by(f32::total_cmp);
    let n = ratios.len();
    (
        ratios[n / 2] / std::f32::consts::LN_2,
        (ratios[3 * n / 4] - ratios[n / 4]) / 3f32.ln(),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spectral_median_and_iqr_recover_exponential_noise_scale() {
        let shape = vec![2.5; PSD_N];
        let p: Vec<_> = (0..PSD_N)
            .map(|i| -((1. - (i as f32 + 0.5) / PSD_N as f32).ln()) * 2.5 * 3.)
            .collect();
        let (m, q) = spectral_scale(&p, &shape);
        assert!((m - 3.).abs() < 0.01);
        assert!((q - 3.).abs() < 0.02);
    }
    #[test]
    fn finite_notches_have_zero_dc_and_nyquist_response() {
        let mut taps = whitening_taps(&vec![1.; PSD_N]);
        notch_offsets(&mut taps);
        let dc = taps.iter().copied().sum::<Complex32>();
        let ny = taps
            .iter()
            .enumerate()
            .map(|(j, v)| *v * if j % 2 == 0 { 1. } else { -1. })
            .sum::<Complex32>();
        assert!(dc.norm() < 1e-5);
        assert!(ny.norm() < 1e-5);
        assert!(taps[0].norm() < 1e-8 && taps[taps.len() - 1].norm() < 1e-8);
    }
    #[test]
    fn offsets_removed_without_touching_an_earlier_echo() {
        let mut z: Vec<_> = (0..4096)
            .map(|j| {
                Complex32::new(7., -3.) + Complex32::new(2., 4.) * if j % 2 == 0 { 1. } else { -1. }
            })
            .collect();
        z[200] += Complex32::new(10., 0.);
        let a = remove_offsets(&mut z);
        assert!((a[0] - Complex32::new(7., -3.)).norm() < 1e-6);
        assert!((a[1] - Complex32::new(2., 4.)).norm() < 1e-6);
        assert!((z[200].re - 10.).abs() < 1e-6);
        assert!(z[3000].norm() < 1e-6);
    }
    #[test]
    fn flat_spectrum_is_identity_filter() {
        let taps = whitening_taps(&vec![1.; PSD_N]);
        assert!((taps[128].re - 1.).abs() < 1e-6);
        assert!(taps
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 128)
            .all(|(_, v)| v.norm() < 1e-6));
    }
}
