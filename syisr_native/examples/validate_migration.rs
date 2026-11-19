// Deterministic noisy moving-echo recovery, with independent pulse phases.
#[path = "../src/filter.rs"]
#[allow(dead_code)]
mod filter;
use anyhow::{ensure, Result};
use num_complex::Complex32;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: validate_migration OUTPUT.h5");
    let mut state = 42871u64;
    let mut uniform = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 32) as f64 + 0.5) / 4294967296.
    };
    let mut rows = Vec::new();
    for down in [true, false] {
        for fd in [-200000., -100000., 100000., 200000.] {
            for snr_db in [20., 30., 60.] {
                for trial in 0..32 {
                    let amp = (10f64.powf(snr_db / 10.) / 800.).sqrt();
                    let mut estimated = Vec::new();
                    let mut data = Vec::new();
                    let mut truths = Vec::new();
                    let mut f = filter::Filter::new(2400, down);
                    let fraction = uniform();
                    for pulse in 0..12 {
                        let t = pulse as f64 * 0.005;
                        let lag = 800. + fraction - fd / 430e6 * t * filter::FS;
                        let phase = uniform() * std::f64::consts::TAU;
                        let z: Vec<_> = (0..2400)
                            .map(|j| {
                                let radius = (-uniform().ln()).sqrt();
                                let phi = std::f64::consts::TAU * uniform();
                                let mut z = Complex32::new(
                                    (radius * phi.cos()) as f32,
                                    (radius * phi.sin()) as f32,
                                );
                                let dt = (j as f64 - lag) / filter::FS;
                                if (0.0..200e-6).contains(&dt) {
                                    let angle = phase
                                        + std::f64::consts::TAU
                                            * ((if down { 1. } else { -1. })
                                                * (2e6 * dt - 1e10 * dt * dt)
                                                + fd * j as f64 / filter::FS);
                                    z += Complex32::from_polar(amp as f32, angle as f32);
                                }
                                z
                            })
                            .collect();
                        let y = f.apply(&z);
                        let k = (600..1100)
                            .max_by(|a, b| y[*a].norm_sqr().total_cmp(&y[*b].norm_sqr()))
                            .unwrap();
                        estimated.push(k as f64);
                        data.push(z);
                        truths.push(lag);
                    }
                    let mt = 0.0275;
                    let ml = estimated.iter().sum::<f64>() / 12.;
                    let slope = estimated
                        .iter()
                        .enumerate()
                        .map(|(j, k)| (j as f64 * 0.005 - mt) * (k - ml))
                        .sum::<f64>()
                        / (0..12)
                            .map(|j| (j as f64 * 0.005 - mt).powi(2))
                            .sum::<f64>();
                    let recovered_fd = -slope / filter::FS * 430e6;
                    let mut e0 = 0.;
                    let mut ec = 0.;
                    let mut p0 = 0.;
                    let mut pc = 0.;
                    for (i, z) in data.iter().enumerate() {
                        let y = f.apply(z);
                        let k = estimated[i] as usize;
                        p0 += y[k].norm_sqr() as f64;
                        e0 += (k as f64 - truths[i]).powi(2);
                        let shifted: Vec<_> = z
                            .iter()
                            .enumerate()
                            .map(|(j, z)| {
                                *z * Complex32::from_polar(
                                    1.,
                                    (-std::f64::consts::TAU * recovered_fd * j as f64 / filter::FS)
                                        as f32,
                                )
                            })
                            .collect();
                        let yc = f.apply(&shifted);
                        let kc = (600..1100)
                            .max_by(|a, b| yc[*a].norm_sqr().total_cmp(&yc[*b].norm_sqr()))
                            .unwrap();
                        ec += (kc as f64 - truths[i]).powi(2);
                        pc += yc[kc].norm_sqr() as f64;
                    }
                    let m_per_sample = filter::C / (2. * filter::FS);
                    rows.push([
                        if down { 1. } else { -1. },
                        fd,
                        snr_db,
                        trial as f64,
                        recovered_fd,
                        (e0 / 12.).sqrt() * m_per_sample,
                        (ec / 12.).sqrt() * m_per_sample,
                        10. * (pc / p0).log10(),
                    ]);
                }
            }
        }
    }
    let max_fd = rows.iter().map(|r| (r[4] - r[1]).abs()).fold(0., f64::max);
    let max_range = rows.iter().map(|r| r[6]).fold(0., f64::max);
    ensure!(max_fd < 5000., "migration Doppler error {max_fd}");
    ensure!(max_range < 30., "corrected range RMS {max_range}");
    ensure!(rows.iter().all(|r| r[6] < r[5]), "range correction failed");
    let h = hdf5::File::create(path)?;
    let flat: Vec<_> = rows.iter().flatten().copied().collect();
    h.new_dataset::<f64>()
        .shape([rows.len(), 8])
        .create("trials")?
        .write_raw(&flat)?;
    h.new_dataset_builder().with_data(b"chirp_sign,doppler_truth_hz,snr_db,trial,doppler_recovered_hz,uncorrected_range_rms_m,corrected_range_rms_m,power_gain_db").create("columns_utf8")?;
    println!("{} noisy moving-echo trials; maximum Doppler error {:.1} Hz; maximum corrected range RMS {:.2} m",rows.len(),max_fd,max_range);
    for snr in [20., 30., 60.] {
        let r: Vec<_> = rows.iter().filter(|r| r[2] == snr).collect();
        println!(
            "SNR {snr:.0} dB: mean uncorrected/corrected RMS {:.2}/{:.2} m; mean gain {:.3} dB",
            r.iter().map(|r| r[5]).sum::<f64>() / r.len() as f64,
            r.iter().map(|r| r[6]).sum::<f64>() / r.len() as f64,
            r.iter().map(|r| r[7]).sum::<f64>() / r.len() as f64
        );
    }
    Ok(())
}
