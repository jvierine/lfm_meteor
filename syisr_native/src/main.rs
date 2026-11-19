mod config;
mod filter;
mod noise;
mod pipeline;
#[cfg(feature = "plots")]
mod plots;
mod raw;
mod timing;
mod track;

use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use config::Config;
use filter::{Filter, C, FS, PULSE_SAMPLES};
use hdf5::{File, Group, H5Type};
#[cfg(feature = "mpi")]
use mpi::traits::*;
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
#[command(
    version,
    about = "Native SYISR station LFM detection, continuous event cuts and noise products"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Write a station configuration; input glob must be quoted in the shell.
    Init {
        #[arg(long,value_parser=["sanya","danzhou","wenchang"])]
        station: String,
        #[arg(long)]
        input: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Validate and describe a few frames without processing the whole recording.
    Inspect {
        file: PathBuf,
        #[arg(long, default_value_t = 8)]
        clock_offset: i32,
        #[arg(long, default_value_t = 8)]
        trailer_bytes: usize,
        #[arg(long, default_value_t = 3)]
        frames: usize,
    },
    /// MPI builds distribute input files; rank zero merges events across files.
    Process {
        #[arg(short, long)]
        config: PathBuf,
        #[arg(long)]
        max_files: Option<usize>,
        #[arg(long)]
        max_frames: Option<usize>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        event_output: Option<PathBuf>,
        #[arg(long)]
        no_plots: bool,
        /// Only calculate noise; skip FFTs, detection and event extraction.
        #[arg(long)]
        noise_only: bool,
        /// Reuse verified completed per-file products after interruption.
        #[arg(long)]
        resume: bool,
        /// Only after confirming numerical compatibility of completed products.
        #[arg(long)]
        allow_build_change: bool,
    },
    /// Re-render event HDF5 files without reading raw recordings.
    Plot { events: PathBuf },
}

#[derive(Clone, Debug)]
struct Pulse {
    utc: i64,
    file: u64,
    offset: u64,
    gate: u64,
    snr: f64,
    noise: f64,
    delay: f64,
    ipp: f64,
    az: f64,
    el: f64,
    gate_front: f64,
    raw_noise: f64,
    raw_mean: f64,
    noise_samples: u64,
    rxaz: f64,
    rxel: f64,
    timing_valid: u8,
}
fn dataset<T: H5Type>(g: &Group, name: &str, v: &[T]) -> Result<()> {
    g.new_dataset::<T>()
        .shape(v.len())
        .create(name)?
        .write_raw(v)?;
    Ok(())
}
fn textdata(g: &Group, name: &str, s: &str) -> Result<()> {
    dataset(g, name, s.as_bytes())
}
fn metadata(f: &File, c: &Config) -> Result<()> {
    textdata(f, "config_toml_utf8", &toml::to_string(c)?)?;
    textdata(f, "program_version_utf8", env!("CARGO_PKG_VERSION"))?;
    textdata(
        f,
        "git_revision_utf8",
        option_env!("SYISR_GIT_REVISION").unwrap_or("unknown"),
    )?;
    textdata(f, "schema_utf8", "syisr-station-v2")?;
    textdata(f, "station_utf8", &c.station)?;
    textdata(f, "range_coordinate_utf8", &c.range_coordinate)?;
    textdata(f,"timing_quality_utf8","timing_valid selects a maximum strictly-increasing subsequence of original header timestamps within each file. Rejected timestamps are retained with timing_valid=0, excluded from detection and sky-noise averages; timestamps are never reconstructed.")?;
    dataset(f, "range_km_per_delay_us", &[c.km_per_us()])?;
    let instrument = f.create_group("instrument")?;
    dataset(&instrument, "sample_rate_hz", &[FS])?;
    dataset(&instrument, "carrier_frequency_hz", &[430_000_000.0f64])?;
    dataset(&instrument, "bandwidth_hz", &[4_000_000.0f64])?;
    dataset(&instrument, "pulse_duration_us", &[200.0f64])?;
    dataset(&instrument, "reference_energy", &[PULSE_SAMPLES as f64])?;
    dataset(
        &instrument,
        "chirp_rate_hz_per_s",
        &[if c.chirp_direction == "down" {
            -2e10f64
        } else {
            2e10f64
        }],
    )?;
    textdata(f,"conventions_utf8","UTC nanoseconds since Unix epoch; voltage=Q+iI; y[k]=sum z[k+j]*conj(s[j]); delay_us=header_gate_front_us+k/fs+delay_correction_us; monostatic range=c*delay/2; bistatic total_path=c*delay; neither is altitude. Positive Doppler shifts a downchirp peak to larger delay. SNR=10log10(peak power/median search-window power). Raw noise=median(|z|^2)/ln(2), ADC counts squared per complex sample; assumes circular Gaussian receiver noise. No Doppler or range-ambiguity correction. Cuts include all available pulses, independent of detection; actual missing frames are flagged, not interpolated.")?;
    Ok(())
}
fn pulses_write(g: &Group, p: &[Pulse]) -> Result<()> {
    macro_rules! col {
        ($name:literal,$field:ident) => {
            dataset(g, $name, &p.iter().map(|x| x.$field).collect::<Vec<_>>())?
        };
    }
    col!("utc_ns", utc);
    col!("source_file_index", file);
    col!("frame_byte_offset", offset);
    col!("echo_start_sample", gate);
    col!("peak_snr_db", snr);
    col!("median_noise_power", noise);
    col!("apparent_delay_us", delay);
    col!("ipp_us", ipp);
    col!("tx_azimuth_deg", az);
    col!("tx_elevation_deg", el);
    col!("gate_front_us", gate_front);
    col!("raw_noise_power_adc2", raw_noise);
    col!("raw_mean_power_adc2", raw_mean);
    col!("noise_sample_count", noise_samples);
    col!("rx_azimuth_deg", rxaz);
    col!("rx_elevation_deg", rxel);
    col!("timing_valid", timing_valid);
    dataset(
        g,
        "total_path_length_km",
        &p.iter().map(|x| x.delay * 1e-9 * C).collect::<Vec<_>>(),
    )?;
    Ok(())
}
fn pulses_read(g: &Group) -> Result<Vec<Pulse>> {
    let utc = g.dataset("utc_ns")?.read_raw::<i64>()?;
    macro_rules! read {
        ($n:literal,$t:ty) => {
            g.dataset($n)?.read_raw::<$t>()?
        };
    }
    let file = read!("source_file_index", u64);
    let offset = read!("frame_byte_offset", u64);
    let gate = read!("echo_start_sample", u64);
    let snr = read!("peak_snr_db", f64);
    let noise = read!("median_noise_power", f64);
    let delay = read!("apparent_delay_us", f64);
    let ipp = read!("ipp_us", f64);
    let az = read!("tx_azimuth_deg", f64);
    let el = read!("tx_elevation_deg", f64);
    let gate_front = read!("gate_front_us", f64);
    let raw_noise = read!("raw_noise_power_adc2", f64);
    let raw_mean = read!("raw_mean_power_adc2", f64);
    let noise_samples = read!("noise_sample_count", u64);
    let rxaz = read!("rx_azimuth_deg", f64);
    let rxel = read!("rx_elevation_deg", f64);
    // Earlier v2 products passed strict monotonicity checks before being written.
    let timing_valid = if g.link_exists("timing_valid") {
        read!("timing_valid", u8)
    } else {
        vec![1; utc.len()]
    };
    Ok((0..utc.len())
        .map(|i| Pulse {
            utc: utc[i],
            file: file[i],
            offset: offset[i],
            gate: gate[i],
            snr: snr[i],
            noise: noise[i],
            delay: delay[i],
            ipp: ipp[i],
            az: az[i],
            el: el[i],
            gate_front: gate_front[i],
            raw_noise: raw_noise[i],
            raw_mean: raw_mean[i],
            noise_samples: noise_samples[i],
            rxaz: rxaz[i],
            rxel: rxel[i],
            timing_valid: timing_valid[i],
        })
        .collect())
}
fn part_path(c: &Config, i: usize) -> PathBuf {
    let name = format!("pulses_{i:06}.h5");
    let path = c.output_dir.join("work").join(&name);
    let legacy = c.output_dir.join(name);
    if path.exists() || !legacy.exists() {
        path
    } else {
        legacy
    }
}
fn modified_ns(path: &Path) -> Result<u64> {
    Ok(path
        .metadata()?
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos()
        .try_into()?)
}
fn verify_config(f: &File, c: &Config, allow_build_change: bool) -> Result<()> {
    let mut old: Config = toml::from_str(&String::from_utf8(
        f.dataset("config_toml_utf8")?.read_raw::<u8>()?,
    )?)?;
    let mut current = c.clone();
    old.make_plots = false;
    current.make_plots = false;
    old.plot_half_width_km = 1.;
    current.plot_half_width_km = 1.;
    ensure!(
        toml::to_string(&old)? == toml::to_string(&current)?,
        "resume configuration differs from saved processing settings"
    );
    let revision = String::from_utf8(f.dataset("git_revision_utf8")?.read_raw::<u8>()?)?;
    ensure!(allow_build_change || revision==option_env!("SYISR_GIT_REVISION").unwrap_or("unknown"),"resume requires the same build revision; use the original executable or explicitly confirm compatible products with --allow-build-change");
    Ok(())
}
fn reusable(c: &Config, path: &Path, index: usize, allow_build_change: bool) -> Result<bool> {
    let part = part_path(c, index);
    if !part.exists() {
        return Ok(false);
    }
    let f = File::open(part)?;
    verify_config(&f, c, allow_build_change)?;
    ensure!(
        String::from_utf8(f.dataset("source_path_utf8")?.read_raw::<u8>()?)?
            == path.to_string_lossy(),
        "resume source manifest changed at file {index}"
    );
    ensure!(
        f.dataset("source_size_bytes")?.read_raw::<u64>()?[0] == path.metadata()?.len()
            && f.dataset("source_mtime_ns")?.read_raw::<u64>()?[0] == modified_ns(path)?,
        "source file changed: {}",
        path.display()
    );
    eprintln!("file {index}: reusing verified completed product");
    Ok(true)
}

fn process_file(c: &Config, path: &Path, index: usize) -> Result<usize> {
    std::fs::create_dir_all(c.output_dir.join("work"))?;
    let started = Instant::now();
    let mut r = raw::Reader::open(path, c.clock_utc_offset_hours, c.trailer_bytes)?;
    r.max_resync_bytes = c.max_resync_bytes;
    r.allow_orphaned_tail = c.allow_orphaned_tail;
    let mut mf = None;
    let mut nprev = 0;
    let mut prev_time = None;
    let mut gap_offsets = Vec::new();
    let mut gap_ns = Vec::new();
    let mut pulses = Vec::new();
    let mut header_hash = Sha256::new();
    while c.max_frames_per_file == 0 || pulses.len() < c.max_frames_per_file {
        let Some(fr) = r.next()? else { break };
        let h = &fr.header;
        if let Some(t) = prev_time {
            if (h.utc_ns - t) as f64 > h.ipp_us * 1500.0 {
                gap_offsets.push(fr.offset);
                gap_ns.push(h.utc_ns - t);
            }
        }
        prev_time = Some(h.utc_ns);
        header_hash.update(fr.header_bytes);
        let (peak, noise, snr, delay) = if c.noise_only {
            (usize::MAX, f64::NAN, f64::NAN, f64::NAN)
        } else {
            if h.n != nprev {
                mf = Some(Filter::new(h.n, c.chirp_direction == "down"));
                nprev = h.n;
            }
            let y = mf.as_mut().unwrap().apply(&fr.iq);
            let lag = |range_km: f64| {
                (range_km / c.km_per_us() - h.gate_front_us - c.delay_correction_us) * FS / 1e6
            };
            let lo = lag(c.min_range_km).ceil().max(0.0) as usize;
            let hi = (lag(c.max_range_km).floor() as isize).min(y.len() as isize - 1);
            ensure!(
                hi >= lo as isize,
                "search range outside recorded valid correlation lags"
            );
            let hi = hi as usize;
            let mut power: Vec<f32> = y[lo..=hi].iter().map(|z| z.norm_sqr()).collect();
            ensure!(
                power.iter().all(|x| x.is_finite()),
                "non-finite matched-filter power"
            );
            let peak = (lo..=hi)
                .max_by(|&a, &b| y[a].norm_sqr().total_cmp(&y[b].norm_sqr()))
                .unwrap();
            let noise = filter::median(&mut power) as f64;
            let snr = if noise > 0. {
                10.0 * (y[peak].norm_sqr() as f64 / noise).log10()
            } else {
                f64::NAN
            };
            (
                peak,
                noise,
                snr,
                h.gate_front_us + c.delay_correction_us + peak as f64 / FS * 1e6,
            )
        };
        let nlo = c
            .noise_min_delay_us
            .map(|v| {
                ((v - h.gate_front_us - c.delay_correction_us) * FS / 1e6)
                    .ceil()
                    .max(0.) as usize
            })
            .unwrap_or(0);
        let nhi = c
            .noise_max_delay_us
            .map(|v| {
                ((v - h.gate_front_us - c.delay_correction_us) * FS / 1e6)
                    .ceil()
                    .max(0.) as usize
            })
            .unwrap_or(fr.iq.len())
            .min(fr.iq.len());
        ensure!(
            nhi > nlo && nhi - nlo >= 32,
            "noise interval outside recorded gate or less than 32 samples"
        );
        let mut powers: Vec<f32> = fr.iq[nlo..nhi].iter().map(|z| z.norm_sqr()).collect();
        let raw_mean = powers.iter().map(|x| *x as f64).sum::<f64>() / powers.len() as f64;
        let raw_noise = filter::median(&mut powers) as f64 / std::f64::consts::LN_2;
        pulses.push(Pulse {
            utc: h.utc_ns,
            file: index as u64,
            offset: fr.offset,
            gate: peak as u64,
            snr,
            noise,
            delay,
            ipp: h.ipp_us,
            az: h.az_deg,
            el: h.el_deg,
            gate_front: h.gate_front_us,
            raw_noise,
            raw_mean,
            noise_samples: (nhi - nlo) as u64,
            rxaz: h.rx_az_deg,
            rxel: h.rx_el_deg,
            timing_valid: 1,
        });
    }
    timing::mark(&mut pulses);
    let rejected: Vec<_> = pulses
        .iter()
        .filter(|p| p.timing_valid == 0)
        .map(|p| p.offset)
        .collect();
    if !rejected.is_empty() {
        eprintln!("WARNING: file {index}: {} temporally inconsistent frames quarantined (original headers retained)",rejected.len());
    }
    let dest = part_path(c, index);
    let temp = dest.with_extension("h5.partial");
    let f = File::create(&temp)?;
    metadata(&f, c)?;
    textdata(&f, "source_path_utf8", &path.to_string_lossy())?;
    dataset(&f, "source_size_bytes", &[path.metadata()?.len()])?;
    dataset(&f, "source_mtime_ns", &[modified_ns(path)?])?;
    textdata(
        &f,
        "processed_header_sha256_utf8",
        &format!("{:x}", header_hash.finalize()),
    )?;
    dataset(
        &f,
        "recovery_byte_offsets",
        &r.recoveries.iter().map(|r| r.0).collect::<Vec<_>>(),
    )?;
    dataset(
        &f,
        "recovery_skipped_bytes",
        &r.recoveries.iter().map(|r| r.1).collect::<Vec<_>>(),
    )?;
    dataset(&f, "timing_gap_before_frame_byte", &gap_offsets)?;
    dataset(&f, "timing_gap_ns", &gap_ns)?;
    dataset(
        &f,
        "orphaned_tail_byte_offset",
        &r.orphaned_tail.map(|v| vec![v.0]).unwrap_or_default(),
    )?;
    dataset(
        &f,
        "orphaned_tail_byte_count",
        &r.orphaned_tail.map(|v| vec![v.1]).unwrap_or_default(),
    )?;
    dataset(
        &f,
        "truncated_trailer_missing_bytes",
        &[r.truncated_trailer_missing_bytes],
    )?;
    dataset(&f, "timing_rejected_frame_byte_offsets", &rejected)?;
    pulses_write(&f.create_group("pulses")?, &pulses)?;
    f.flush()?;
    drop(f);
    std::fs::rename(temp, dest)?;
    eprintln!(
        "file {index}: {} pulses in {:.2} s ({})",
        pulses.len(),
        started.elapsed().as_secs_f64(),
        path.display()
    );
    Ok(pulses.len())
}

fn group_events(c: &Config, pulses: Vec<Pulse>) -> (Vec<Vec<Pulse>>, usize) {
    let mut events = Vec::new();
    let mut current: Vec<Pulse> = Vec::new();
    let mut long = 0;
    for p in pulses {
        if !p.snr.is_finite() || p.snr < c.snr_threshold_db {
            continue;
        }
        if let Some(last) = current.last() {
            let dt = (p.utc - last.utc) as f64 * 1e-9;
            if dt > c.max_gap_ipps as f64 * last.ipp * 1e-6 + 1e-7
                || (p.az - last.az).abs() > 0.01
                || (p.el - last.el).abs() > 0.01
                || (p.rxaz - last.rxaz).abs() > 0.01
                || (p.rxel - last.rxel).abs() > 0.01
                || p.ipp != last.ipp
            {
                finish_event(c, &mut current, &mut events, &mut long);
            }
        }
        current.push(p);
    }
    finish_event(c, &mut current, &mut events, &mut long);
    (events, long)
}
fn finish_event(
    c: &Config,
    current: &mut Vec<Pulse>,
    events: &mut Vec<Vec<Pulse>>,
    long: &mut usize,
) {
    if current.len() >= c.min_event_pulses {
        let dt = (current.last().unwrap().utc - current[0].utc) as f64 * 1e-9;
        if dt <= c.max_event_duration_s {
            events.push(std::mem::take(current));
        } else {
            *long += 1;
        }
    }
    current.clear();
}
fn write_event(
    c: &Config,
    files: &[PathBuf],
    index: usize,
    p: &[Pulse],
    track: &track::Track,
) -> Result<()> {
    let dest = c.event_dir().join(format!("event_{index:06}.h5"));
    let tmp = dest.with_extension("h5.partial");
    let f = File::create(&tmp)?;
    metadata(&f, c)?;
    pulses_write(&f.create_group("pulses")?, p)?;
    dataset(&f, "sample_rate_hz", &[FS])?;
    let detected: Vec<_> = p
        .iter()
        .map(|p| u8::from(track.pulses.iter().any(|d| d.utc == p.utc)))
        .collect();
    dataset(&f, "ransac_inlier", &detected)?;
    dataset(&f, "track_epoch_utc_ns", &[track.epoch])?;
    dataset(&f, "track_range_at_epoch_km", &[track.intercept])?;
    dataset(&f, "track_range_rate_km_s", &[track.rate])?;
    dataset(&f, "track_rms_km", &[track.rms])?;
    let t0 = track.pulses[0].utc - (c.cut_time_padding_s * 1e9).round() as i64;
    let t1 = track.pulses.last().unwrap().utc + (c.cut_time_padding_s * 1e9).round() as i64;
    dataset(&f, "requested_time_bounds_utc_ns", &[t0, t1])?;
    dataset(
        &f,
        "time_padding_truncated",
        &[
            u8::from(
                p.first()
                    .is_none_or(|x| x.utc > t0 + ((x.ipp * 1000.) as i64)),
            ),
            u8::from(
                p.last()
                    .is_none_or(|x| x.utc < t1 - ((x.ipp * 1000.) as i64)),
            ),
        ],
    )?;
    dataset(
        &f,
        "gap_before_ns",
        &p.iter()
            .enumerate()
            .map(|(i, pulse)| {
                if i > 0 && (pulse.utc - p[i - 1].utc) as f64 > p[i - 1].ipp * 1500. {
                    pulse.utc - p[i - 1].utc
                } else {
                    0
                }
            })
            .collect::<Vec<i64>>(),
    )?;
    let mut raw_words = Vec::new();
    let mut raw_offsets = vec![0u64];
    let mut starts = Vec::new();
    let mut mf_real = Vec::new();
    let mut mf_imag = Vec::new();
    let mut mf_offsets = vec![0u64];
    let mut mf_starts = Vec::new();
    let mut headers = Vec::new();
    let mut clipped = Vec::new();
    let mut active_file = u64::MAX;
    let mut reader = None;
    let mut filter = None;
    let mut nprev = 0;
    for pulse in p {
        if active_file != pulse.file {
            reader = Some(raw::Reader::open(
                &files[pulse.file as usize],
                c.clock_utc_offset_hours,
                c.trailer_bytes,
            )?);
            reader.as_mut().unwrap().allow_orphaned_tail = c.allow_orphaned_tail;
            active_file = pulse.file;
        }
        let r = reader.as_mut().unwrap();
        r.seek(pulse.offset)?;
        let fr = r.next()?.context("missing source frame")?;
        ensure!(
            fr.header.utc_ns == pulse.utc,
            "source changed during extraction"
        );
        let pad = (c.raw_padding_us * FS / 1e6).ceil() as usize;
        let centre = track.range(pulse.utc) / c.km_per_us();
        let lo = ((centre
            - c.cut_range_half_width_km / c.km_per_us()
            - fr.header.gate_front_us
            - c.delay_correction_us)
            * FS
            / 1e6)
            .floor() as i64;
        let hi = ((centre + c.cut_range_half_width_km / c.km_per_us()
            - fr.header.gate_front_us
            - c.delay_correction_us)
            * FS
            / 1e6)
            .ceil() as i64;
        let max_lag = (fr.iq.len() - PULSE_SAMPLES) as i64;
        clipped.push(u8::from(lo < 0 || hi > max_lag));
        ensure!(
            hi >= 0 && lo <= max_lag,
            "event track/padding lies completely outside recorder gate"
        );
        let a = lo.max(0) as usize;
        let b = (hi.min(max_lag) as usize) + 1;
        let start = a.saturating_sub(pad);
        let stop = (b - 1 + PULSE_SAMPLES + pad).min(fr.iq.len());
        raw_words.extend_from_slice(&fr.words[start * 2..stop * 2]);
        raw_offsets.push(raw_words.len() as u64 / 2);
        starts.push(start as u64);
        headers.extend_from_slice(&fr.header_bytes);
        if nprev != fr.iq.len() {
            filter = Some(Filter::new(fr.iq.len(), c.chirp_direction == "down"));
            nprev = fr.iq.len();
        }
        let y = filter.as_mut().unwrap().apply(&fr.iq);
        mf_real.extend(y[a..b].iter().map(|z| z.re));
        mf_imag.extend(y[a..b].iter().map(|z| z.im));
        mf_offsets.push(mf_real.len() as u64);
        mf_starts.push(a as u64);
    }
    let g = f.create_group("raw")?;
    // Ragged cuts: offsets index IQ pairs, words alternate recorder I then Q.
    dataset(&g, "iq_words_iq", &raw_words)?;
    dataset(&g, "pulse_offsets", &raw_offsets)?;
    dataset(&g, "first_sample", &starts)?;
    dataset(
        &g,
        "first_delay_us",
        &p.iter()
            .zip(&starts)
            .map(|(p, s)| p.gate_front + c.delay_correction_us + *s as f64 / FS * 1e6)
            .collect::<Vec<_>>(),
    )?;
    dataset(&g, "frame_headers_u8", &headers)?;
    drop(g);
    let g = f.create_group("matched_filter")?;
    dataset(&g, "real", &mf_real)?;
    dataset(&g, "imag", &mf_imag)?;
    dataset(&g, "pulse_offsets", &mf_offsets)?;
    dataset(&g, "first_lag_sample", &mf_starts)?;
    dataset(
        &g,
        "first_delay_us",
        &p.iter()
            .zip(&mf_starts)
            .map(|(p, s)| p.gate_front + c.delay_correction_us + *s as f64 / FS * 1e6)
            .collect::<Vec<_>>(),
    )?;
    drop(g);
    dataset(&f, "range_padding_truncated", &clipped)?;
    dataset(
        &f,
        "predicted_track_range_km",
        &p.iter().map(|p| track.range(p.utc)).collect::<Vec<_>>(),
    )?;
    textdata(
        &f,
        "source_files_utf8",
        &files
            .iter()
            .map(|x| x.to_string_lossy())
            .collect::<Vec<_>>()
            .join("\n"),
    )?;
    f.flush()?;
    f.close()?;
    std::fs::rename(tmp, dest)?;
    if c.make_plots {
        #[cfg(feature = "plots")]
        plots::render(&c.event_dir().join(format!("event_{index:06}.h5")))
            .with_context(|| format!("rendering event {index:06}"))?;
    }
    Ok(())
}
fn finalize(
    c: &Config,
    files: &[PathBuf],
    counts: &[u64],
    detections: usize,
    events: &[track::Track],
    long: usize,
) -> Result<()> {
    let tmp = c.output_dir.join("catalogue.h5.partial");
    let f = File::create(&tmp)?;
    metadata(&f, c)?;
    textdata(
        &f,
        "source_files_utf8",
        &files
            .iter()
            .map(|x| x.to_string_lossy())
            .collect::<Vec<_>>()
            .join("\n"),
    )?;
    dataset(&f, "processed_pulses_per_file", counts)?;
    dataset(&f, "threshold_crossings", &[detections as u64])?;
    dataset(&f, "overlong_window_rejections", &[long as u64])?;
    dataset(
        &f,
        "event_pulse_counts",
        &events
            .iter()
            .map(|t| t.pulses.len() as u64)
            .collect::<Vec<_>>(),
    )?;
    dataset(
        &f,
        "event_start_utc_ns",
        &events.iter().map(|t| t.pulses[0].utc).collect::<Vec<_>>(),
    )?;
    dataset(
        &f,
        "event_stop_utc_ns",
        &events
            .iter()
            .map(|t| t.pulses.last().unwrap().utc)
            .collect::<Vec<_>>(),
    )?;
    dataset(
        &f,
        "event_peak_snr_db",
        &events
            .iter()
            .map(|t| {
                t.pulses
                    .iter()
                    .map(|p| p.snr)
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect::<Vec<_>>(),
    )?;
    dataset(
        &f,
        "event_median_apparent_range_km",
        &events
            .iter()
            .map(|t| {
                let mut r: Vec<_> = t
                    .pulses
                    .iter()
                    .map(|p| (p.delay * c.km_per_us()) as f32)
                    .collect();
                filter::median(&mut r) as f64
            })
            .collect::<Vec<_>>(),
    )?;
    f.flush()?;
    drop(f);
    std::fs::rename(tmp, c.output_dir.join("catalogue.h5"))?;
    eprintln!("COMPLETE: {} pulses, {detections} threshold crossings, {} candidate meteor events; {long} overlong event windows rejected",counts.iter().sum::<u64>(),events.len());
    Ok(())
}
fn run(cli: Cli, rank: usize, size: usize, barrier: impl Fn()) -> Result<()> {
    match cli.command {
        Command::Init {
            station,
            input,
            output,
            config,
        } => {
            if rank == 0 {
                use std::io::Write;
                let mut c: Config = toml::from_str(include_str!("../config.toml"))?;
                c.station = match station.as_str() {
                    "sanya" => "Sanya",
                    "danzhou" => "Danzhou",
                    _ => "Wenchang",
                }
                .into();
                if station != "sanya" {
                    c.range_coordinate = "total_path".into();
                    c.min_range_km = 200.;
                    c.max_range_km = 550.;
                    c.max_range_rate_km_s = 200.;
                }
                c.input_glob = input;
                c.output_dir = output;
                c.validate()?;
                let mut f = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&config)?;
                f.write_all(toml::to_string_pretty(&c)?.as_bytes())?;
                println!("Wrote {}. Review the clock offset and hardware-delay calibration, then run: syisr-native process --config {}",config.display(),config.display());
            }
        }
        Command::Inspect {
            file,
            clock_offset,
            trailer_bytes,
            frames,
        } => {
            if rank == 0 {
                let mut r = raw::Reader::open(&file, clock_offset, trailer_bytes)?;
                for _ in 0..frames {
                    let Some(f) = r.next()? else { break };
                    println!("byte={} UTC={} samples={} gate_front={} us IPP={} us pulse={} us az={:.3} el={:.3}",f.offset,chrono::DateTime::from_timestamp_nanos(f.header.utc_ns),f.header.n,f.header.gate_front_us,f.header.ipp_us,f.header.pulse_us,f.header.az_deg,f.header.el_deg);
                }
            }
        }
        Command::Process {
            config,
            max_files,
            max_frames,
            output,
            event_output,
            no_plots,
            noise_only,
            resume,
            allow_build_change,
        } => {
            let mut c = Config::load(&config)?;
            if let Some(n) = max_files {
                c.max_files = n;
            }
            if let Some(n) = max_frames {
                c.max_frames_per_file = n;
            }
            if let Some(p) = output {
                c.output_dir = p;
            }
            if let Some(p) = event_output {
                c.event_dir = Some(p);
            }
            if no_plots {
                c.make_plots = false;
            }
            if noise_only {
                c.noise_only = true;
            }
            if c.noise_only {
                c.make_plots = false;
            }
            c.validate()?;
            if c.make_plots {
                #[cfg(feature = "plots")]
                plots::check_tools()?;
                #[cfg(not(feature = "plots"))]
                anyhow::bail!("plot support not compiled; rebuild normally or use --no-plots");
            }
            let files = c.files()?;
            if rank == 0 {
                if c.output_dir.exists() {
                    ensure!(resume,"output already exists: {}; use --resume with the same configuration and executable",c.output_dir.display());
                    verify_config(
                        &File::open(c.output_dir.join("run.h5"))?,
                        &c,
                        allow_build_change,
                    )?;
                } else {
                    ensure!(
                        !c.event_dir().exists(),
                        "event output already exists: {}",
                        c.event_dir().display()
                    );
                }
                std::fs::create_dir_all(&c.output_dir)?;
                std::fs::create_dir_all(c.output_dir.join("work"))?;
                std::fs::create_dir_all(c.event_dir())?;
                if !c.output_dir.join("run.h5").exists() {
                    let f = File::create(c.output_dir.join("run.h5"))?;
                    metadata(&f, &c)?;
                    f.flush()?;
                }
                eprintln!(
                    "{} inputs, {size} ranks; output {}",
                    files.len(),
                    c.output_dir.display()
                );
            }
            barrier();
            if size == 1 {
                let mut live = pipeline::Live::new(&c, &files);
                for (i, p) in files.iter().enumerate() {
                    if !(resume && reusable(&c, p, i, allow_build_change)?) {
                        process_file(&c, p, i)?;
                    }
                    live.poll()?;
                }
                live.finish()?;
            } else if rank == 0 {
                let mut live = pipeline::Live::new(&c, &files);
                while !live.poll()? {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
                live.finish()?;
            } else {
                for (i, p) in files
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % (size - 1) == rank - 1)
                {
                    if resume && reusable(&c, p, i, allow_build_change)? {
                        continue;
                    }
                    process_file(&c, p, i).with_context(|| format!("{}", p.display()))?;
                }
            }
            barrier();
        }
        Command::Plot { events } => {
            if rank == 0 {
                #[cfg(feature = "plots")]
                plots::directory(&events)?;
                #[cfg(not(feature = "plots"))]
                {
                    let _ = events;
                    anyhow::bail!("rebuild with plots feature");
                }
            }
        }
    }
    Ok(())
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    #[cfg(feature = "mpi")]
    {
        let universe = mpi::initialize().context("MPI already initialized")?;
        let world = universe.world();
        if let Err(e) = run(cli, world.rank() as usize, world.size() as usize, || {
            world.barrier()
        }) {
            eprintln!("rank {}: {e:#}", world.rank());
            world.abort(1);
        }
        Ok(())
    }
    #[cfg(not(feature = "mpi"))]
    {
        for key in [
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "PMIX_SIZE",
            "SLURM_NTASKS",
        ] {
            if let Ok(s) = std::env::var(key) {
                ensure!(
                    s.parse::<usize>().unwrap_or(1) <= 1,
                    "serial build under parallel launcher: build with --features mpi"
                );
            }
        }
        run(cli, 0, 1, || {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn event_crosses_file_boundary_and_rejects_jump() {
        let mut c = Config::load(Path::new("config.toml")).unwrap();
        c.min_event_pulses = 6;
        let mut p = Vec::new();
        for i in 0..8 {
            p.push(Pulse {
                utc: i * 5_000_000,
                file: (i / 4) as u64,
                offset: 0,
                gate: 0,
                snr: 30.0,
                noise: 1.0,
                delay: 650.0,
                ipp: 5000.0,
                az: 15.0,
                el: 75.0,
                gate_front: 466.,
                raw_noise: 1.,
                raw_mean: 1.,
                noise_samples: 1000,
                rxaz: 15.,
                rxel: 75.,
                timing_valid: 1,
            });
        }
        let (e, l) = group_events(&c, p.clone());
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].len(), 8);
        assert_eq!(l, 0);
        p[4].delay = 900.0;
        let (groups, _) = group_events(&c, p);
        let fit = track::fit(&c, groups.into_iter().next().unwrap()).unwrap();
        assert_eq!(fit.pulses.len(), 7);
    }
    #[test]
    fn hdf5_empty_and_populated_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let f = File::create(dir.path().join("t.h5")).unwrap();
        let g = f.create_group("empty").unwrap();
        pulses_write(&g, &[]).unwrap();
        assert!(pulses_read(&g).unwrap().is_empty());
    }
    #[test]
    fn real_format_end_to_end_with_cross_file_event() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let mut c = Config::load(Path::new("config.toml")).unwrap();
        c.input_glob = dir.path().join("*.dat").to_string_lossy().into_owned();
        c.output_dir = dir.path().join("products");
        c.make_plots = false;
        #[cfg(feature = "plots")]
        {
            c.make_plots = plots::check_tools().is_ok();
        }
        let s = filter::chirp(true);
        for file in 0..2 {
            let mut f = std::fs::File::create(dir.path().join(format!("raw-{file}.dat"))).unwrap();
            for k in 0..8 {
                let pulse = file * 8 + k;
                f.write_all(&raw::fixture_header(4000, pulse * 200))
                    .unwrap();
                for j in 0..4000 {
                    let echo = if (4..12).contains(&pulse) && pulse != 7 && (800..1600).contains(&j)
                    {
                        s[j - 800] * 1000.0
                    } else {
                        num_complex::Complex32::default()
                    };
                    let re = echo.re as i32 + ((j * 71 + k as usize * 13) % 23) as i32 - 11;
                    let im = echo.im as i32 + ((j * 37 + k as usize * 17) % 19) as i32 - 9;
                    f.write_all(&im.to_le_bytes()).unwrap();
                    f.write_all(&re.to_le_bytes()).unwrap();
                }
                f.write_all(&[0u8; 8]).unwrap();
            }
        }
        let config = dir.path().join("run.toml");
        std::fs::write(&config, toml::to_string(&c).unwrap()).unwrap();
        let cli = Cli {
            command: Command::Process {
                config,
                max_files: None,
                max_frames: None,
                output: None,
                event_output: None,
                no_plots: false,
                noise_only: false,
                resume: false,
                allow_build_change: false,
            },
        };
        run(cli, 0, 1, || {}).unwrap();
        let f = File::open(c.output_dir.join("catalogue.h5")).unwrap();
        assert_eq!(
            f.dataset("event_pulse_counts")
                .unwrap()
                .read_raw::<u64>()
                .unwrap(),
            vec![7]
        );
        let f = File::open(c.event_dir().join("event_000000.h5")).unwrap();
        let cuts = pulses_read(&f.group("pulses").unwrap()).unwrap();
        assert_eq!(cuts.len(), 16);
        assert_eq!(cuts[7].utc - cuts[6].utc, 5_000_000);
        assert!(cuts.iter().all(|p| p.gate_front == 466.));
        let inliers = f
            .dataset("ransac_inlier")
            .unwrap()
            .read_raw::<u8>()
            .unwrap();
        assert_eq!(inliers.iter().sum::<u8>(), 7);
        assert_eq!(inliers[7], 0);
        let offsets = f
            .dataset("raw/pulse_offsets")
            .unwrap()
            .read_raw::<u64>()
            .unwrap();
        assert_eq!(offsets.len(), 17);
        assert!(offsets.windows(2).all(|v| v[1] > v[0]));
        let words = f
            .dataset("raw/iq_words_iq")
            .unwrap()
            .read_raw::<i32>()
            .unwrap();
        assert_eq!(words.len(), offsets[16] as usize * 2);
        let first = f
            .dataset("raw/first_sample")
            .unwrap()
            .read_raw::<u64>()
            .unwrap();
        let delay = f
            .dataset("raw/first_delay_us")
            .unwrap()
            .read_raw::<f64>()
            .unwrap();
        assert!(first
            .iter()
            .zip(&delay)
            .all(|(s, d)| (*d - (466. + *s as f64 / 4.)).abs() < 1e-10));
        let n = File::open(c.output_dir.join("noise.h5")).unwrap();
        if c.make_plots {
            assert!(c.event_dir().join("plots/event_000000.png").exists());
        }
        assert_eq!(n.dataset("pulses/utc_ns").unwrap().size(), 16);
        let inputs = c.files().unwrap();
        assert!(reusable(&c, &inputs[0], 0, false).unwrap());
        let mut different = c.clone();
        different.snr_threshold_db += 1.;
        assert!(reusable(&different, &inputs[0], 0, false).is_err());
        std::fs::OpenOptions::new()
            .append(true)
            .open(&inputs[0])
            .unwrap()
            .write_all(&[0])
            .unwrap();
        assert!(reusable(&c, &inputs[0], 0, false).is_err());
    }
    #[test]
    fn init_remote_and_delay_coordinates() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("danzhou.toml");
        run(
            Cli {
                command: Command::Init {
                    station: "danzhou".into(),
                    input: dir.path().join("*.dat").to_string_lossy().into_owned(),
                    output: dir.path().join("products"),
                    config: cfg.clone(),
                },
            },
            0,
            1,
            || {},
        )
        .unwrap();
        let c = Config::load(&cfg).unwrap();
        assert_eq!(c.range_coordinate, "total_path");
        let tau = 466. + 800. / 4.;
        let path = tau * c.km_per_us();
        assert!((path - 199.661777028).abs() < 1e-9);
        assert!(((path / c.km_per_us() - 466.) * 4. - 800.).abs() < 1e-9);
    }
    #[test]
    fn publishes_raw_event_before_last_input_is_processed() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let mut c = Config::load(Path::new("config.toml")).unwrap();
        c.input_glob = dir.path().join("*.dat").to_string_lossy().into_owned();
        c.output_dir = dir.path().join("out");
        c.make_plots = false;
        c.max_event_duration_s = 0.1;
        c.cut_time_padding_s = 0.01;
        std::fs::create_dir_all(c.event_dir()).unwrap();
        let code = filter::chirp(true);
        for file in 0..3 {
            let mut f = std::fs::File::create(dir.path().join(format!("raw-{file}.dat"))).unwrap();
            for k in 0..50 {
                f.write_all(&raw::fixture_header(4000, (file * 50 + k) * 200))
                    .unwrap();
                for j in 0..4000 {
                    let z = if file == 0 && (20..31).contains(&k) && (800..1600).contains(&j) {
                        code[j - 800] * 1000.
                    } else {
                        num_complex::Complex32::default()
                    };
                    let a = z.im as i32 + ((j * 37 + k as usize * 17) % 19) as i32 - 9;
                    let b = z.re as i32 + ((j * 71 + k as usize * 13) % 23) as i32 - 11;
                    f.write_all(&a.to_le_bytes()).unwrap();
                    f.write_all(&b.to_le_bytes()).unwrap();
                }
                f.write_all(&[0; 8]).unwrap();
            }
        }
        let files = c.files().unwrap();
        let mut live = pipeline::Live::new(&c, &files);
        process_file(&c, &files[0], 0).unwrap();
        assert!(!live.poll().unwrap());
        process_file(&c, &files[1], 1).unwrap();
        assert!(!live.poll().unwrap());
        assert!(c.event_dir().join("event_000000.h5").exists());
        assert!(!part_path(&c, 2).exists());
        assert!(!c.output_dir.join("catalogue.h5").exists());
        process_file(&c, &files[2], 2).unwrap();
        assert!(live.poll().unwrap());
        live.finish().unwrap();
    }
}
