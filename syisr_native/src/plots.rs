use crate::config::Config;
use anyhow::{ensure, Context, Result};
use rustplotlib::*;
use std::{path::Path, process::Command};
pub fn render_complete(path: &Path) -> bool {
    use std::io::{Read, Seek, SeekFrom};
    (|| -> std::io::Result<bool> {
        let mut f = std::fs::File::open(path)?;
        let mut head = [0; 8];
        let mut tail = [0; 12];
        f.read_exact(&mut head)?;
        f.seek(SeekFrom::End(-12))?;
        f.read_exact(&mut tail)?;
        Ok(head == [137, 80, 78, 71, 13, 10, 26, 10]
            && tail == [0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130])
    })()
    .unwrap_or(false)
}

pub fn check_tools() -> Result<()> {
    for tool in ["latex", "dvisvgm"] {
        ensure!(
            Command::new(tool)
                .arg("--version")
                .output()
                .with_context(|| format!(
                    "{tool} is required for plots; install TeX or use --no-plots"
                ))?
                .status
                .success(),
            "{tool} is unavailable"
        );
    }
    Ok(())
}
pub fn directory(dir: &Path) -> Result<()> {
    check_tools()?;
    let mut files = std::fs::read_dir(dir)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.retain(|p| {
        p.extension().is_some_and(|s| s == "h5")
            && p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("event_")
    });
    files.sort();
    for file in files {
        render(&file)?;
    }
    Ok(())
}
pub fn render(path: &Path) -> Result<()> {
    let h = hdf5::File::open(path).with_context(|| format!("opening event {}", path.display()))?;
    let c: Config = toml::from_str(&String::from_utf8(
        h.dataset("config_toml_utf8")?.read_raw::<u8>()?,
    )?)?;
    let utc = h.dataset("pulses/utc_ns")?.read_raw::<i64>()?;
    if utc.is_empty() {
        return Ok(());
    }
    let read = |n: &str| h.dataset(n)?.read_raw::<f64>();
    let noise = read("pulses/median_noise_power")?;
    let gate = read("pulses/gate_front_us")?;
    let track = read("predicted_track_range_km")?;
    let epoch = h.dataset("track_epoch_utc_ns")?.read_raw::<i64>()?[0];
    let ipp = read("pulses/ipp_us")?[0] * 0.001;
    let offsets = h
        .dataset("matched_filter/pulse_offsets")?
        .read_raw::<u64>()?;
    let first = h
        .dataset("matched_filter/first_lag_sample")?
        .read_raw::<u64>()?;
    let re = h.dataset("matched_filter/real")?.read_raw::<f32>()?;
    let im = h.dataset("matched_filter/imag")?.read_raw::<f32>()?;
    let t: Vec<_> = utc.iter().map(|n| (*n - epoch) as f64 * 1e-6).collect();
    let dx = c.km_per_us() / 4.;
    let rmin = track.iter().copied().fold(f64::INFINITY, f64::min) - c.plot_half_width_km;
    let rmax = track.iter().copied().fold(f64::NEG_INFINITY, f64::max) + c.plot_half_width_km;
    let ny = ((rmax - rmin) / dx).ceil() as usize;
    let nx = ((t.last().unwrap() - t[0]) / ipp).round() as usize + 1;
    ensure!(nx * ny < 50_000_000, "plot grid too large");
    let mut values = vec![f64::NAN; nx * ny];
    for i in 0..utc.len() {
        let x = ((t[i] - t[0]) / ipp).round() as usize;
        for y in 0..ny {
            let range = rmin + (y as f64 + 0.5) * dx;
            let lag =
                ((range / c.km_per_us() - gate[i] - c.delay_correction_us) * 4.).round() as i64;
            let j = lag - first[i] as i64;
            if j >= 0 && j < (offsets[i + 1] - offsets[i]) as i64 {
                let k = offsets[i] as usize + j as usize;
                values[y * nx + x] = 10.
                    * (((re[k] as f64).powi(2) + (im[k] as f64).powi(2)).max(1e-30) / noise[i])
                        .log10();
            }
        }
    }
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 150.;
    f.height_mm = 103.;
    let max = h
        .dataset("pulses/peak_snr_db")?
        .read_raw::<f64>()?
        .into_iter()
        .fold(25., f64::max)
        .ceil();
    let scale = f.scale(Scale::new(
        [0., max],
        Normalization::Linear,
        ColorMap::sequential(),
    )?);
    let mut p = Panel::new(
        Rect::new(24., 11., 100., 74.),
        Axes::new(
            [t[0] - ipp / 2., t[0] + nx as f64 * ipp - ipp / 2.],
            [rmin, rmin + ny as f64 * dx],
            "Time (ms)",
            c.range_label(),
        ),
    );
    p.push(Layer::Pcolormesh(Field::new(
        (0..=nx).map(|i| t[0] + i as f64 * ipp - ipp / 2.).collect(),
        (0..=ny).map(|i| rmin + i as f64 * dx).collect(),
        values,
        scale,
    )?));
    f.panels.push(p);
    f.colorbars.push(Colorbar::new(
        Rect::new(126., 11., 3., 74.),
        scale,
        "SNR (dB)",
        vec![0., 10., 20.],
    ));
    let tm = chrono::DateTime::from_timestamp_nanos(epoch);
    f.labels.push(Label::new(
        format!("{}  {} UTC", c.station, tm.format("%Y-%m-%d %H:%M:%S%.3f")),
        76.,
        5.,
    ));
    let dir = path.parent().unwrap().join("plots");
    std::fs::create_dir_all(&dir)?;
    let staging = dir.join(".rendering");
    std::fs::create_dir_all(&staging)?;
    let name = Path::new(path.file_stem().unwrap()).with_extension("png");
    let temp = staging.join(&name);
    f.save(&temp)?;
    std::fs::rename(temp, dir.join(name))?;
    Ok(())
}
