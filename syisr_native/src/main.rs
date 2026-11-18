mod config;
mod filter;
mod raw;

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
    about = "Native Sanya monostatic 4 MHz LFM detection and voltage cuts"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
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
    },
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
    textdata(f, "schema_utf8", "syisr-monostatic-v1")?;
    textdata(f,"conventions_utf8","UTC nanoseconds since Unix epoch; voltage=Q+iI; y[k]=sum z[k+j]*conj(s[j]); delay_us is echo-start gate plus explicit calibration; apparent monostatic range=c*delay/2, NOT altitude; positive Doppler shifts a downchirp matched-filter peak to larger delay; SNR=10log10(peak power/median search-window power); no Doppler or range-ambiguity correction; IQ cuts retain signed int32 recorder words")?;
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
    dataset(
        g,
        "apparent_monostatic_range_km",
        &p.iter()
            .map(|x| x.delay * 1e-6 * C / 2000.0)
            .collect::<Vec<_>>(),
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
        })
        .collect())
}
fn part_path(c: &Config, i: usize) -> PathBuf {
    c.output_dir.join(format!("pulses_{i:06}.h5"))
}

fn process_file(c: &Config, path: &Path, index: usize) -> Result<usize> {
    let started = Instant::now();
    let mut r = raw::Reader::open(path, c.clock_utc_offset_hours, c.trailer_bytes)?;
    r.max_resync_bytes = c.max_resync_bytes;
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
            ensure!(
                h.utc_ns > t,
                "non-increasing timestamps at byte {}",
                fr.offset
            );
            if (h.utc_ns - t) as f64 > h.ipp_us * 1500.0 {
                gap_offsets.push(fr.offset);
                gap_ns.push(h.utc_ns - t);
            }
        }
        prev_time = Some(h.utc_ns);
        header_hash.update(fr.header_bytes);
        if h.n != nprev {
            mf = Some(Filter::new(h.n, c.chirp_direction == "down"));
            nprev = h.n;
        }
        let y = mf.as_mut().unwrap().apply(&fr.iq);
        let lag = |range_km: f64| {
            (range_km * 2000.0 / C * 1e6 - h.gate_front_us - c.delay_correction_us) * FS / 1e6
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
        ensure!(
            noise > 0.0,
            "zero matched-filter noise floor at byte {}",
            fr.offset
        );
        let snr = 10.0 * (y[peak].norm_sqr() as f64 / noise).log10();
        pulses.push(Pulse {
            utc: h.utc_ns,
            file: index as u64,
            offset: fr.offset,
            gate: peak as u64,
            snr,
            noise,
            delay: h.gate_front_us + c.delay_correction_us + peak as f64 / FS * 1e6,
            ipp: h.ipp_us,
            az: h.az_deg,
            el: h.el_deg,
        });
    }
    let dest = part_path(c, index);
    let temp = dest.with_extension("h5.partial");
    let f = File::create(&temp)?;
    metadata(&f, c)?;
    textdata(&f, "source_path_utf8", &path.to_string_lossy())?;
    dataset(&f, "source_size_bytes", &[path.metadata()?.len()])?;
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
        if p.snr < c.snr_threshold_db {
            continue;
        }
        if let Some(last) = current.last() {
            let dt = (p.utc - last.utc) as f64 * 1e-9;
            let dr = (p.delay - last.delay).abs() * 1e-6 * C / 2000.0;
            if dt > c.max_gap_ipps as f64 * last.ipp * 1e-6 + 1e-7
                || dr > c.range_slack_km + c.max_range_rate_km_s * dt
                || (p.az - last.az).abs() > 0.01
                || (p.el - last.el).abs() > 0.01
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
fn write_event(c: &Config, files: &[PathBuf], index: usize, p: &[Pulse]) -> Result<()> {
    let dest = c
        .output_dir
        .join("events")
        .join(format!("event_{index:06}.h5"));
    let tmp = dest.with_extension("h5.partial");
    let f = File::create(&tmp)?;
    metadata(&f, c)?;
    pulses_write(&f.create_group("pulses")?, p)?;
    let mut raw_words = Vec::new();
    let mut raw_offsets = vec![0u64];
    let mut starts = Vec::new();
    let mut mf_real = Vec::new();
    let mut mf_imag = Vec::new();
    let mut mf_offsets = vec![0u64];
    let mut mf_starts = Vec::new();
    let mut headers = Vec::new();
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
        let gate = pulse.gate as usize;
        let start = gate.saturating_sub(pad);
        let stop = (gate + PULSE_SAMPLES + pad).min(fr.iq.len());
        raw_words.extend_from_slice(&fr.words[start * 2..stop * 2]);
        raw_offsets.push(raw_words.len() as u64 / 2);
        starts.push(start as u64);
        headers.extend_from_slice(&fr.header_bytes);
        if nprev != fr.iq.len() {
            filter = Some(Filter::new(fr.iq.len(), c.chirp_direction == "down"));
            nprev = fr.iq.len();
        }
        let y = filter.as_mut().unwrap().apply(&fr.iq);
        let half = (c.matched_filter_half_width_us * FS / 1e6).ceil() as usize;
        let a = gate.saturating_sub(half);
        let b = (gate + half + 1).min(y.len());
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
    dataset(&g, "frame_headers_u8", &headers)?;
    let g = f.create_group("matched_filter")?;
    dataset(&g, "real", &mf_real)?;
    dataset(&g, "imag", &mf_imag)?;
    dataset(&g, "pulse_offsets", &mf_offsets)?;
    dataset(&g, "first_lag_sample", &mf_starts)?;
    dataset(&f, "sample_rate_hz", &[FS])?;
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
    drop(f);
    std::fs::rename(tmp, dest)?;
    Ok(())
}
fn merge(c: &Config, files: &[PathBuf]) -> Result<()> {
    let mut pulses = Vec::new();
    let mut counts = Vec::new();
    for i in 0..files.len() {
        let f = File::open(part_path(c, i))?;
        let rows = pulses_read(&f.group("pulses")?)?;
        counts.push(rows.len() as u64);
        pulses.extend(rows.into_iter().filter(|p| p.snr >= c.snr_threshold_db));
    }
    pulses.sort_by_key(|p| (p.utc, p.file, p.offset));
    ensure!(
        pulses.windows(2).all(|p| p[0].utc != p[1].utc),
        "overlapping input timestamps in detections; remove overlapping recordings"
    );
    let detections = pulses.len();
    let (events, long) = group_events(c, pulses);
    for (i, p) in events.iter().enumerate() {
        write_event(c, files, i, p)?;
    }
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
    dataset(&f, "processed_pulses_per_file", &counts)?;
    dataset(&f, "threshold_crossings", &[detections as u64])?;
    dataset(&f, "rejected_overlong_tracks", &[long as u64])?;
    dataset(
        &f,
        "event_pulse_counts",
        &events.iter().map(|p| p.len() as u64).collect::<Vec<_>>(),
    )?;
    dataset(
        &f,
        "event_start_utc_ns",
        &events.iter().map(|p| p[0].utc).collect::<Vec<_>>(),
    )?;
    dataset(
        &f,
        "event_stop_utc_ns",
        &events
            .iter()
            .map(|p| p.last().unwrap().utc)
            .collect::<Vec<_>>(),
    )?;
    dataset(
        &f,
        "event_peak_snr_db",
        &events
            .iter()
            .map(|p| p.iter().map(|p| p.snr).fold(f64::NEG_INFINITY, f64::max))
            .collect::<Vec<_>>(),
    )?;
    dataset(
        &f,
        "event_median_apparent_range_km",
        &events
            .iter()
            .map(|p| {
                let mut r: Vec<_> = p
                    .iter()
                    .map(|p| (p.delay * 1e-6 * C / 2000.0) as f32)
                    .collect();
                filter::median(&mut r) as f64
            })
            .collect::<Vec<_>>(),
    )?;
    f.flush()?;
    drop(f);
    std::fs::rename(tmp, c.output_dir.join("catalogue.h5"))?;
    eprintln!("COMPLETE: {} pulses, {detections} threshold crossings, {} candidate meteor events; {long} overlong tracks rejected",counts.iter().sum::<u64>(),events.len());
    Ok(())
}
fn run(cli: Cli, rank: usize, size: usize, barrier: impl Fn()) -> Result<()> {
    match cli.command {
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
            c.validate()?;
            let files = c.files()?;
            if rank == 0 {
                ensure!(
                    !c.output_dir.exists(),
                    "output already exists: {}; choose a new directory",
                    c.output_dir.display()
                );
                std::fs::create_dir_all(c.output_dir.join("events"))?;
                eprintln!(
                    "{} inputs, {size} ranks; output {}",
                    files.len(),
                    c.output_dir.display()
                );
            }
            barrier();
            for (i, p) in files.iter().enumerate().filter(|(i, _)| i % size == rank) {
                process_file(&c, p, i).with_context(|| format!("{}", p.display()))?;
            }
            barrier();
            if rank == 0 {
                merge(&c, &files)?;
            }
            barrier();
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
            });
        }
        let (e, l) = group_events(&c, p.clone());
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].len(), 8);
        assert_eq!(l, 0);
        p[4].delay = 900.0;
        assert!(group_events(&c, p).0.is_empty());
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
        let s = filter::chirp(true);
        for file in 0..2 {
            let mut f = std::fs::File::create(dir.path().join(format!("raw-{file}.dat"))).unwrap();
            for k in 0..4 {
                f.write_all(&raw::fixture_header(4000, (file * 4 + k) * 200))
                    .unwrap();
                for j in 0..4000 {
                    let echo = if (800..1600).contains(&j) {
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
            },
        };
        run(cli, 0, 1, || {}).unwrap();
        let f = File::open(c.output_dir.join("catalogue.h5")).unwrap();
        assert_eq!(
            f.dataset("event_pulse_counts")
                .unwrap()
                .read_raw::<u64>()
                .unwrap(),
            vec![8]
        );
        let f = File::open(c.output_dir.join("events/event_000000.h5")).unwrap();
        assert!(pulses_read(&f.group("pulses").unwrap())
            .unwrap()
            .iter()
            .all(|p| p.gate == 800));
        let offsets = f
            .dataset("raw/pulse_offsets")
            .unwrap()
            .read_raw::<u64>()
            .unwrap();
        assert_eq!(offsets, (0..=8).map(|i| i * 1200).collect::<Vec<_>>());
        let words = f
            .dataset("raw/iq_words_iq")
            .unwrap()
            .read_raw::<i32>()
            .unwrap();
        assert_eq!(words.len(), 8 * 1200 * 2);
    }
}
