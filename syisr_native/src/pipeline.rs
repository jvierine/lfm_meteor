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
    reconcile: bool,
    last_utc: Option<i64>,
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
            reconcile: false,
            last_utc: None,
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
            if rows
                .first()
                .zip(self.last_utc)
                .is_some_and(|(p, last)| p.utc <= last)
            {
                if !self.reconcile {
                    eprintln!("INPUT OVERLAP: switching to verified chronological reconciliation after scanning; existing previews will be archived");
                }
                self.reconcile = true;
            }
            if let Some(last) = rows.last() {
                self.last_utc = Some(self.last_utc.unwrap_or(last.utc).max(last.utc));
            }
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
        if self.reconcile {
            self.cache.clear();
            self.owner = self.loaded;
            return Ok(());
        }
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
            if window.windows(2).any(|p| p[0].utc == p[1].utc) {
                self.reconcile = true;
                eprintln!(
                    "INPUT OVERLAP: deferring extraction to verified chronological reconciliation"
                );
                self.cache.clear();
                self.owner = self.loaded;
                return Ok(());
            }
            let (groups, long) = group_events(self.c, window.clone());
            let candidates = groups.len();
            let mut fit_rejections = 0;
            self.long += long;
            for group in groups {
                let Some(t) = track::fit(self.c, group) else {
                    fit_rejections += 1;
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
            eprintln!("EXTRACT file {}: through {} UTC; {} candidate windows, {} fit rejections, {} events published",self.owner-1,chrono::DateTime::from_timestamp_nanos(last),candidates,fit_rejections,self.events.len());
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
        if self.reconcile {
            return self.finish_reconciled();
        }
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

    fn finish_reconciled(mut self) -> Result<()> {
        use crate::{dataset, raw};
        use std::collections::HashMap;
        let mut rows = Vec::new();
        for i in 0..self.files.len() {
            let f = open_index(&part_path(self.c, i))?;
            rows.extend(pulses_read(&f.group("pulses")?)?);
        }
        rows.sort_by_key(|p| (p.utc, p.file, p.offset));
        let mut readers = HashMap::new();
        let mut unique: Vec<Pulse> = Vec::with_capacity(rows.len());
        let mut duplicates = Vec::new();
        for pulse in rows {
            if pulse.timing_valid == 1 {
                if let Some(previous) = unique
                    .iter()
                    .rev()
                    .take_while(|p| p.utc == pulse.utc)
                    .find(|p| p.timing_valid == 1)
                {
                    let mut frames = Vec::new();
                    for p in [previous, &pulse] {
                        if !readers.contains_key(&p.file) {
                            let mut reader = raw::Reader::open(
                                &self.files[p.file as usize],
                                self.c.clock_utc_offset_hours,
                                self.c.trailer_bytes,
                            )?;
                            reader.allow_orphaned_tail = self.c.allow_orphaned_tail;
                            readers.insert(p.file, reader);
                        }
                        let reader = readers.get_mut(&p.file).unwrap();
                        reader.seek(p.offset)?;
                        frames.push(reader.next()?.context("missing duplicate source frame")?);
                    }
                    ensure!(frames[0].header_bytes == frames[1].header_bytes && frames[0].words == frames[1].words,
                        "conflicting voltage at UTC {}: {} byte {} versus {} byte {}; refusing to discard either measurement",
                        pulse.utc, self.files[previous.file as usize].display(), previous.offset, self.files[pulse.file as usize].display(), pulse.offset);
                    duplicates.push([
                        pulse.utc,
                        previous.file as i64,
                        previous.offset as i64,
                        pulse.file as i64,
                        pulse.offset as i64,
                    ]);
                    continue;
                }
            }
            unique.push(pulse);
        }
        let f = hdf5::File::create(self.c.output_dir.join("input_reconciliation.h5"))?;
        let flat: Vec<_> = duplicates.iter().flatten().copied().collect();
        f.new_dataset::<i64>()
            .shape([duplicates.len(), 5])
            .create("duplicate_frames")?
            .write_raw(&flat)?;
        dataset(
            &f,
            "columns_utf8",
            b"utc_ns,kept_file_index,kept_byte_offset,duplicate_file_index,duplicate_byte_offset",
        )?;
        f.close()?;
        eprintln!(
            "RECONCILED: {} unique rows, {} byte-verified duplicate frames",
            unique.len(),
            duplicates.len()
        );
        self.noise = unique.iter().map(noise::NoiseRow::from).collect();
        unique.retain(|p| p.timing_valid == 1);
        self.detections = unique
            .iter()
            .filter(|p| p.snr >= self.c.snr_threshold_db)
            .count();
        let detections = unique
            .iter()
            .filter(|p| p.snr >= self.c.snr_threshold_db)
            .cloned()
            .collect();
        let (groups, long) = group_events(self.c, detections);
        self.long = long;
        self.events.clear();
        if self.c.event_dir().read_dir()?.next().is_some() {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos();
            let archived = self
                .c
                .output_dir
                .join(format!("event_previews_before_reconciliation_{stamp}"));
            std::fs::rename(self.c.event_dir(), &archived)?;
            eprintln!("Preserved previews: {}", archived.display());
        }
        std::fs::create_dir_all(self.c.event_dir())?;
        for group in groups {
            let Some(t) = track::fit(self.c, group) else {
                continue;
            };
            let pad = (self.c.cut_time_padding_s * 1e9).round() as i64;
            let lo = unique.partition_point(|p| p.utc < t.pulses[0].utc - pad);
            let hi = unique.partition_point(|p| p.utc <= t.pulses.last().unwrap().utc + pad);
            write_event(self.c, self.files, self.events.len(), &unique[lo..hi], &t)?;
            self.events.push(t);
        }
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
