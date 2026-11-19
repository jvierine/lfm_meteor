use crate::{config::Config, Pulse};

#[derive(Debug)]
pub struct Track {
    pub pulses: Vec<Pulse>,
    pub epoch: i64,
    pub intercept: f64,
    pub rate: f64,
    pub rms: f64,
}
impl Track {
    pub fn range(&self, t: i64) -> f64 {
        self.intercept + self.rate * (t - self.epoch) as f64 * 1e-9
    }
}
pub fn fit(c: &Config, p: Vec<Pulse>) -> Option<Track> {
    if p.len() < c.min_event_pulses {
        return None;
    }
    let epoch = p[0].utc;
    let x: Vec<_> = p.iter().map(|p| (p.utc - epoch) as f64 * 1e-9).collect();
    let y: Vec<_> = p.iter().map(|p| p.delay * c.km_per_us()).collect();
    let mut best = Vec::new();
    let mut score = f64::INFINITY;
    let mut state = epoch as u64 ^ 0x9e3779b97f4a7c15;
    for k in 0..c.ransac_iterations {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let a = (state >> 32) as usize % p.len();
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let b = if k == 0 {
            p.len() - 1
        } else {
            (state >> 32) as usize % p.len()
        };
        let a = if k == 0 { 0 } else { a };
        if a == b || x[a] == x[b] {
            continue;
        }
        let rate = (y[b] - y[a]) / (x[b] - x[a]);
        if rate.abs() > c.max_range_rate_km_s {
            continue;
        }
        let intercept = y[a] - rate * x[a];
        let ids: Vec<_> = (0..p.len())
            .filter(|&i| (y[i] - intercept - rate * x[i]).abs() <= c.range_slack_km)
            .collect();
        let error = ids
            .iter()
            .map(|&i| (y[i] - intercept - rate * x[i]).powi(2))
            .sum::<f64>();
        if ids.len() > best.len() || (ids.len() == best.len() && error < score) {
            best = ids;
            score = error;
        }
    }
    if best.len() < c.min_event_pulses
        || (best.len() as f64) < c.ransac_min_fraction * p.len() as f64
    {
        return None;
    }
    let mut intercept = 0.;
    let mut rate = 0.;
    for _ in 0..3 {
        let n = best.len() as f64;
        let mx = best.iter().map(|&i| x[i]).sum::<f64>() / n;
        let my = best.iter().map(|&i| y[i]).sum::<f64>() / n;
        let xx = best.iter().map(|&i| (x[i] - mx).powi(2)).sum::<f64>();
        if xx == 0. {
            return None;
        }
        rate = best.iter().map(|&i| (x[i] - mx) * (y[i] - my)).sum::<f64>() / xx;
        intercept = my - rate * mx;
        best = (0..p.len())
            .filter(|&i| (y[i] - intercept - rate * x[i]).abs() <= c.range_slack_km)
            .collect();
        if best.len() < c.min_event_pulses {
            return None;
        }
    }
    if rate.abs() > c.max_range_rate_km_s
        || (best.len() as f64) < c.ransac_min_fraction * p.len() as f64
    {
        return None;
    }
    let rms = (best
        .iter()
        .map(|&i| (y[i] - intercept - rate * x[i]).powi(2))
        .sum::<f64>()
        / best.len() as f64)
        .sqrt();
    let pulses = best.into_iter().map(|i| p[i].clone()).collect();
    Some(Track {
        pulses,
        epoch,
        intercept,
        rate,
        rms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recovers_line_despite_outliers() {
        let c = Config::load(std::path::Path::new("config.toml")).unwrap();
        let p: Vec<_> = (0..20)
            .map(|i| Pulse {
                utc: i * 5_000_000,
                file: 0,
                offset: 0,
                gate: 0,
                snr: 30.,
                noise: 1.,
                delay: (100. - 30. * i as f64 * 0.005 + if i % 5 == 0 { 8. } else { 0. })
                    / c.km_per_us(),
                ipp: 5000.,
                az: 15.,
                el: 75.,
                gate_front: 466.,
                raw_noise: 1.,
                raw_mean: 1.,
                noise_samples: 1000,
                rxaz: 15.,
                rxel: 75.,
            })
            .collect();
        let t = fit(&c, p).unwrap();
        assert_eq!(t.pulses.len(), 16);
        assert!((t.rate + 30.).abs() < 1e-8);
        assert!((t.intercept - 100.).abs() < 1e-8);
    }
}
