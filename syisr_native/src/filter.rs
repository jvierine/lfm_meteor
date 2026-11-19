use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub const FS: f64 = 4_000_000.0;
pub const C: f64 = 299_792_458.0;
pub const PULSE_SAMPLES: usize = 800;
pub fn chirp(down: bool) -> Vec<Complex32> {
    (0..PULSE_SAMPLES)
        .map(|i| {
            let t = i as f64 / FS;
            let phase = std::f64::consts::TAU
                * (2e6 * t - 0.5 * (4e6 / 200e-6) * t * t)
                * if down { 1.0 } else { -1.0 };
            Complex32::from_polar(1.0, phase as f32)
        })
        .collect()
}
pub struct Filter {
    len: usize,
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
    reference: Vec<Complex32>,
    work: Vec<Complex32>,
    scratch: Vec<Complex32>,
}
impl Filter {
    pub fn new(n: usize, down: bool) -> Self {
        let len = (n + PULSE_SAMPLES - 1).next_power_of_two();
        let mut plan = FftPlanner::new();
        let fwd = plan.plan_fft_forward(len);
        let inv = plan.plan_fft_inverse(len);
        let mut reference = vec![Complex32::default(); len];
        reference[..PULSE_SAMPLES].copy_from_slice(&chirp(down));
        fwd.process(&mut reference);
        let scratch = vec![
            Complex32::default();
            fwd.get_inplace_scratch_len()
                .max(inv.get_inplace_scratch_len())
        ];
        Self {
            len,
            fwd,
            inv,
            reference,
            work: vec![Complex32::default(); len],
            scratch,
        }
    }
    // y[k] = sum_j z[k+j] conj(s[j]); k is the echo-envelope start sample.
    pub fn apply(&mut self, z: &[Complex32]) -> &[Complex32] {
        self.work.fill(Complex32::default());
        self.work[..z.len()].copy_from_slice(z);
        self.fwd
            .process_with_scratch(&mut self.work, &mut self.scratch);
        for (v, s) in self.work.iter_mut().zip(&self.reference) {
            *v *= s.conj() / self.len as f32;
        }
        self.inv
            .process_with_scratch(&mut self.work, &mut self.scratch);
        &self.work[..z.len() - PULSE_SAMPLES + 1]
    }
}
pub fn median(x: &mut [f32]) -> f32 {
    let n = x.len();
    let (_, v, _) = x.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
    let hi = *v;
    if n % 2 == 1 {
        hi
    } else {
        0.5 * (hi + x[..n / 2].iter().copied().max_by(f32::total_cmp).unwrap())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fft_matches_direct_correlation() {
        let z: Vec<_> = (0..1000)
            .map(|j| Complex32::new((j as f32 * 0.23).sin(), (j as f32 * 0.37).cos()))
            .collect();
        let mut f = Filter::new(z.len(), true);
        let y = f.apply(&z);
        let s = chirp(true);
        for k in [0, 27, 200] {
            let d: Complex32 = z[k..k + 800]
                .iter()
                .zip(&s)
                .map(|(a, b)| a * b.conj())
                .sum();
            assert!((d - y[k]).norm() < 0.002);
        }
    }
    #[test]
    fn injected_echo_lag_and_amplitude() {
        let mut z = vec![Complex32::default(); 2000];
        let s = chirp(true);
        z[351..1151].copy_from_slice(&s);
        let mut f = Filter::new(z.len(), true);
        let y = f.apply(&z);
        let k = (0..y.len())
            .max_by(|&a, &b| y[a].norm_sqr().total_cmp(&y[b].norm_sqr()))
            .unwrap();
        assert_eq!(k, 351);
        assert!((y[k].re - 800.0).abs() < 0.01);
    }
    #[test]
    fn downchirp_doppler_displaces_peak() {
        let mut z = vec![Complex32::default(); 4000];
        let s = chirp(true);
        for (j, v) in s.into_iter().enumerate() {
            z[1000 + j] = v * Complex32::from_polar(
                1.0,
                (std::f64::consts::TAU * 100000.0 * j as f64 / FS) as f32,
            );
        }
        let mut f = Filter::new(z.len(), true);
        let y = f.apply(&z);
        let k = (0..y.len())
            .max_by(|&a, &b| y[a].norm_sqr().total_cmp(&y[b].norm_sqr()))
            .unwrap();
        assert_eq!(k, 1020);
        assert!((y[k].norm() / 800.0 - 0.975).abs() < 0.001);
    }
    #[test]
    fn doppler_compensation_recovers_echo_for_both_chirps_and_signs() {
        for down in [true, false] {
            for fd in [-100000., 100000.] {
                let mut z = vec![Complex32::default(); 4000];
                for (j, s) in chirp(down).into_iter().enumerate() {
                    z[1000 + j] = s * Complex32::from_polar(
                        1.,
                        (std::f64::consts::TAU * fd * (1000 + j) as f64 / FS) as f32,
                    );
                }
                for (j, z) in z.iter_mut().enumerate() {
                    *z *= Complex32::from_polar(
                        1.,
                        (-std::f64::consts::TAU * fd * j as f64 / FS) as f32,
                    );
                }
                let mut f = Filter::new(z.len(), down);
                let y = f.apply(&z);
                let k = (0..y.len())
                    .max_by(|a, b| y[*a].norm_sqr().total_cmp(&y[*b].norm_sqr()))
                    .unwrap();
                assert_eq!(k, 1000);
                assert!((y[k].norm() - 800.).abs() < 0.01);
            }
        }
    }
    #[test]
    fn full_delay_doppler_bank_recovers_known_echo() {
        for down in [true, false] {
            for truth_fd in [-100000., 100000.] {
                let mut z = vec![Complex32::default(); 2400];
                for (j, s) in chirp(down).into_iter().enumerate() {
                    z[700 + j] = s * Complex32::from_polar(
                        1.,
                        (std::f64::consts::TAU * truth_fd * (700 + j) as f64 / FS) as f32,
                    );
                }
                let mut f = Filter::new(z.len(), down);
                let mut best = (0., 0., 0);
                for row in 0..=40 {
                    let fd = -200000. + row as f64 * 10000.;
                    let shifted: Vec<_> = z
                        .iter()
                        .enumerate()
                        .map(|(j, z)| {
                            *z * Complex32::from_polar(
                                1.,
                                (-std::f64::consts::TAU * fd * j as f64 / FS) as f32,
                            )
                        })
                        .collect();
                    for (k, y) in f.apply(&shifted).iter().enumerate() {
                        if y.norm_sqr() > best.0 {
                            best = (y.norm_sqr(), fd, k);
                        }
                    }
                }
                assert_eq!(best.1, truth_fd);
                assert_eq!(best.2, 700);
            }
        }
    }
    #[test]
    fn complex_gaussian_noise_power_normalization() {
        let mut state = 713u32;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f64 + 0.5) / 4294967296.0
        };
        let z: Vec<_> = (0..32768)
            .map(|_| {
                let a = (-2.0 * uniform().ln()).sqrt();
                let phi = std::f64::consts::TAU * uniform();
                Complex32::new((a * phi.cos()) as f32, (a * phi.sin()) as f32)
            })
            .collect();
        let mut f = Filter::new(z.len(), true);
        let mut power: Vec<_> = f.apply(&z).iter().map(|v| v.norm_sqr()).collect();
        let mean = power.iter().map(|v| *v as f64).sum::<f64>() / power.len() as f64;
        assert!((mean / 1600.0 - 1.0).abs() < 0.04);
        assert!((median(&mut power) as f64 / (1600.0 * 2f64.ln()) - 1.0).abs() < 0.05);
    }
}
