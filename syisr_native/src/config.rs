use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub input_glob: String,
    pub output_dir: PathBuf,
    pub clock_utc_offset_hours: i32,
    pub trailer_bytes: usize,
    pub max_resync_bytes: usize,
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
    pub matched_filter_half_width_us: f64,
    pub max_files: usize,
    pub max_frames_per_file: usize,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let mut c: Self = toml::from_str(&std::fs::read_to_string(path)?)?;
        let base = path.canonicalize()?.parent().unwrap().to_owned();
        if c.output_dir.is_relative() {
            c.output_dir = base.join(&c.output_dir);
        }
        if Path::new(&c.input_glob).is_relative() {
            c.input_glob = base.join(&c.input_glob).to_string_lossy().into_owned();
        }
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<()> {
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
            self.matched_filter_half_width_us,
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
                && (0.0..=1000.0).contains(&self.matched_filter_half_width_us),
            "invalid cut padding"
        );
        Ok(())
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
