use crate::{
    config::Config, finalize, group_events, noise, open_index, part_path, pulses_read, track,
    write_event, Pulse,
};
use anyhow::{ensure, Context, Result};
use std::{collections::VecDeque, path::PathBuf};

struct Chunk {
    index: usize,
    rows: Vec<Pulse>,
}
pub struct Live<'a> {
    c: &'a Config,
    files: &'a [PathBuf],
    loaded: usize,
    owner: usize,
    cache: VecDeque<Chunk>,
    counts: Vec<u64>,
    noise: Vec<noise::NoiseRow>,
    events: Vec<track::Track>,
    detections: usize,
    long: usize,
}
impl<'a> Live<'a> {
    pub fn new(c: &'a Config, files: &'a [PathBuf]) -> Self {
        Self {
            c,
            files,
            loaded: 0,
            owner: 0,
            cache: VecDeque::new(),
            counts: Vec::new(),
            noise: Vec::new(),
            events: Vec::new(),
            detections: 0,
            long: 0,
        }
    }
    pub fn poll(&mut self) -> Result<bool> {
        while self.loaded < self.files.len() && part_path(self.c, self.loaded).exists() {
            let f = open_index(&part_path(self.c, self.loaded)).with_context(|| {
                format!("reading index {}", part_path(self.c, self.loaded).display())
            })?;
            let mut rows = pulses_read(&f.group("pulses")?)?;
            f.close()?;
            self.counts.push(rows.len() as u64);
            self.noise.extend(rows.iter().map(noise::NoiseRow::from));
            rows.retain(|p| p.timing_valid == 1);
            rows.sort_by_key(|p| p.utc);
            self.detections += rows
                .iter()
                .filter(|p| p.snr >= self.c.snr_threshold_db)
                .count();
            self.cache.push_back(Chunk {
                index: self.loaded,
                rows,
            });
            self.loaded += 1;
            self.emit_ready()?;
        }
        Ok(self.loaded == self.files.len())
    }
    fn emit_ready(&mut self) -> Result<()> {
        if self.c.noise_only {
            self.owner = self.loaded;
            self.cache.clear();
            return Ok(());
        }
        let look = ((self.c.max_event_duration_s + self.c.cut_time_padding_s) * 1e9).round() as i64;
        while self.owner < self.loaded {
            let central = self.cache.iter().find(|p| p.index == self.owner).unwrap();
            if central.rows.is_empty() {
                self.owner += 1;
                continue;
            }
            let first = central.rows[0].utc;
            let last = central.rows.last().unwrap().utc;
            let gap = (central.rows.iter().map(|p| p.ipp).fold(0., f64::max)
                * self.c.max_gap_ipps as f64
                * 1000.)
                .ceil() as i64;
            let newest = self
                .cache
                .iter()
                .rev()
                .find_map(|p| p.rows.last())
                .map(|p| p.utc)
                .unwrap();
            if self.loaded < self.files.len() && newest < last + look + gap {
                break;
            }
            let lo = first - look - gap;
            let hi = last + look + gap;
            let mut window: Vec<_> = self
                .cache
                .iter()
                .flat_map(|p| &p.rows)
                .filter(|p| p.utc >= lo && p.utc <= hi)
                .cloned()
                .collect();
            window.sort_by_key(|p| p.utc);
            ensure!(
                window.windows(2).all(|p| p[0].utc != p[1].utc),
                "overlapping valid UTCs in event window"
            );
            let (groups, long) = group_events(self.c, window.clone());
            self.long += long;
            for group in groups {
                let Some(t) = track::fit(self.c, group) else {
                    continue;
                };
                if t.pulses[0].file != self.owner as u64 {
                    continue;
                }
                let pad = (self.c.cut_time_padding_s * 1e9).round() as i64;
                let a = t.pulses[0].utc - pad;
                let b = t.pulses.last().unwrap().utc + pad;
                let cuts: Vec<_> = window
                    .iter()
                    .filter(|p| p.utc >= a && p.utc <= b)
                    .cloned()
                    .collect();
                let index = self.events.len();
                write_event(self.c, self.files, index, &cuts, &t)?;
                eprintln!(
                    "EVENT {index:06}: {} raw-voltage pulses saved, {} detection inliers; plot {}",
                    cuts.len(),
                    t.pulses.len(),
                    self.c
                        .event_dir()
                        .join(format!("plots/event_{index:06}.png"))
                        .display()
                );
                self.events.push(t);
            }
            self.owner += 1;
            let next_first = self
                .cache
                .iter()
                .find(|p| p.index == self.owner)
                .and_then(|p| p.rows.first())
                .map(|p| p.utc);
            if let Some(t) = next_first {
                while self.cache.front().is_some_and(|p| {
                    p.index < self.owner
                        && p.rows.last().is_none_or(|last| last.utc < t - look - gap)
                }) {
                    self.cache.pop_front();
                }
            }
        }
        Ok(())
    }
    pub fn finish(mut self) -> Result<()> {
        ensure!(
            self.loaded == self.files.len(),
            "cannot finalize incomplete input scan"
        );
        self.emit_ready()?;
        noise::write(self.c, &mut self.noise)?;
        finalize(
            self.c,
            self.files,
            &self.counts,
            self.detections,
            &self.events,
            self.long,
        )
    }
}
