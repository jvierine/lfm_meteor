use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub station: String,
    pub range_coordinate: String,
    pub input_glob: String,
    pub output_dir: PathBuf,
    pub event_dir: Option<PathBuf>,
    pub clock_utc_offset_hours: i32,
    pub trailer_bytes: usize,
    pub max_resync_bytes: usize,
    #[serde(default = "default_true")]
    pub allow_orphaned_tail: bool,
    pub chirp_direction: String,
    pub delay_correction_us: f64,
    pub delay_calibration: String,
    pub min_range_km: f64,
    pub max_range_km: f64,
    pub snr_threshold_db: f64,
    pub max_gap_ipps: usize,
    pub min_event_pulses: usize,
    pub max_range_rate_km_s: f64,
    pub range_slack_km: f64,
    pub max_event_duration_s: f64,
    pub raw_padding_us: f64,
    pub cut_range_half_width_km: f64,
    pub cut_time_padding_s: f64,
    pub ransac_iterations: usize,
    pub ransac_min_fraction: f64,
    pub noise_min_delay_us: Option<f64>,
    pub noise_max_delay_us: Option<f64>,
    pub noise_average_pulses: usize,
    pub make_plots: bool,
    #[serde(default)]
    pub noise_only: bool,
    pub plot_half_width_km: f64,
    pub max_files: usize,
    pub max_frames_per_file: usize,
}
fn default_true() -> bool {
    true
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let mut c: Self = toml::from_str(&std::fs::read_to_string(path)?)?;
        let base = path.canonicalize()?.parent().unwrap().to_owned();
        if c.output_dir.is_relative() {
            c.output_dir = base.join(&c.output_dir);
        }
        if let Some(p) = &mut c.event_dir {
            if p.is_relative() {
                *p = base.join(&*p);
            }
        }
        if Path::new(&c.input_glob).is_relative() {
            c.input_glob = base.join(&c.input_glob).to_string_lossy().into_owned();
        }
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.station.is_empty(), "station name is required");
        ensure!(
            matches!(self.range_coordinate.as_str(), "monostatic" | "total_path"),
            "range_coordinate must be monostatic or total_path"
        );
        ensure!(
            self.noise_average_pulses > 0 && self.ransac_iterations > 0,
            "invalid noise/RANSAC count"
        );
        ensure!(
            (0.0..=1.0).contains(&self.ransac_min_fraction),
            "invalid RANSAC fraction"
        );
        for v in [self.noise_min_delay_us, self.noise_max_delay_us]
            .into_iter()
            .flatten()
        {
            ensure!(v.is_finite() && v >= 0., "invalid noise delay interval");
        }
        if let (Some(a), Some(b)) = (self.noise_min_delay_us, self.noise_max_delay_us) {
            ensure!(b > a, "invalid noise delay bounds");
        }
        ensure!(
            (-23..=23).contains(&self.clock_utc_offset_hours),
            "invalid clock offset"
        );
        ensure!(self.trailer_bytes <= 1024, "unreasonable trailer size");
        ensure!(
            self.max_resync_bytes <= 16_000_000,
            "resync window exceeds 16 MB"
        );
        ensure!(
            matches!(self.chirp_direction.as_str(), "up" | "down"),
            "chirp_direction must be up or down"
        );
        for x in [
            self.delay_correction_us,
            self.min_range_km,
            self.max_range_km,
            self.snr_threshold_db,
            self.max_range_rate_km_s,
            self.range_slack_km,
            self.max_event_duration_s,
            self.raw_padding_us,
            self.cut_range_half_width_km,
            self.cut_time_padding_s,
            self.plot_half_width_km,
        ] {
            ensure!(x.is_finite(), "non-finite configuration value");
        }
        ensure!(
            self.min_range_km >= 0.0 && self.max_range_km > self.min_range_km,
            "invalid range interval"
        );
        ensure!(
            self.min_event_pulses >= 2 && self.max_gap_ipps >= 1,
            "invalid event grouping"
        );
        ensure!(
            self.max_range_rate_km_s > 0.0
                && self.range_slack_km >= 0.0
                && self.max_event_duration_s > 0.0,
            "invalid track limits"
        );
        ensure!(
            (0.0..=1000.0).contains(&self.raw_padding_us)
                && self.cut_range_half_width_km > 0.
                && self.cut_time_padding_s >= 0.
                && self.plot_half_width_km > 0.
                && self.plot_half_width_km <= self.cut_range_half_width_km,
            "invalid cut padding"
        );
        Ok(())
    }
    pub fn event_dir(&self) -> PathBuf {
        self.event_dir
            .clone()
            .unwrap_or_else(|| self.output_dir.join("event_cuts"))
    }
    pub fn km_per_us(&self) -> f64 {
        299_792_458.0 * 1e-9
            / if self.range_coordinate == "monostatic" {
                2.
            } else {
                1.
            }
    }
    #[cfg(feature = "plots")]
    pub fn range_label(&self) -> &str {
        if self.range_coordinate == "monostatic" {
            "Apparent range (km)"
        } else {
            "Apparent path length (km)"
        }
    }
    pub fn files(&self) -> Result<Vec<PathBuf>> {
        let mut files =
            glob::glob(&self.input_glob)?.collect::<std::result::Result<Vec<_>, _>>()?;
        files.retain(|p| p.is_file());
        // Numeric suffix order within each acquisition prefix (not 1, 10, 100, 2).
        files.sort_by_key(|p| {
            let s = p.to_string_lossy();
            match s.rsplit_once('-') {
                Some((a, b)) => (a.to_owned(), b.parse::<u64>().unwrap_or(u64::MAX)),
                None => (s.into_owned(), 0),
            }
        });
        if self.max_files > 0 {
            files.truncate(self.max_files);
        }
        if files.is_empty() {
            bail!("no inputs match {}", self.input_glob);
        }
        for p in &files {
            p.metadata()
                .with_context(|| format!("input {}", p.display()))?;
        }
        Ok(files)
    }
}
