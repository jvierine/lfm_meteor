use anyhow::Result;
use rustplotlib::*;
use std::path::PathBuf;

fn main() -> Result<()> {
    let root = PathBuf::from(std::env::args().nth(1).expect("event directory"));
    let out = root.join("previews");
    std::fs::create_dir_all(&out)?;
    let mut files = std::fs::read_dir(&root)?
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
    for (event, path) in files.iter().enumerate() {
        let h = hdf5::File::open(path)?;
        let read = |s: &str| h.dataset(s).unwrap().read_raw::<f64>().unwrap();
        let utc = h.dataset("pulses/utc_ns")?.read_raw::<i64>()?;
        let snr = read("pulses/peak_snr_db");
        let noise = read("pulses/median_noise_power");
        let delay = read("pulses/apparent_delay_us");
        let gate = h.dataset("pulses/echo_start_sample")?.read_raw::<u64>()?;
        let offsets = h
            .dataset("matched_filter/pulse_offsets")?
            .read_raw::<u64>()?;
        let first = h
            .dataset("matched_filter/first_lag_sample")?
            .read_raw::<u64>()?;
        let re = h.dataset("matched_filter/real")?.read_raw::<f32>()?;
        let im = h.dataset("matched_filter/imag")?.read_raw::<f32>()?;
        let t: Vec<_> = utc.iter().map(|n| (*n - utc[0]) as f64 * 1e-6).collect();
        let ranges: Vec<_> = delay.iter().map(|d| d * 0.149896229).collect();
        let rmin = ranges.iter().copied().fold(f64::INFINITY, f64::min) - 1.;
        let rmax = ranges.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 1.;
        let dx = 0.03747405725;
        let ny = ((rmax - rmin) / dx).ceil() as usize;
        let nx = (t.last().unwrap() / 5.).round() as usize + 1;
        let mut values = vec![f64::NAN; nx * ny];
        for i in 0..utc.len() {
            let x = (t[i] / 5.).round() as usize;
            for y in 0..ny {
                let r = rmin + (y as f64 + 0.5) * dx;
                let lag = gate[i] as i64 + ((r - ranges[i]) / dx).round() as i64;
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
        f.width_mm = 190.;
        f.height_mm = 88.;
        let scale = f.scale(Scale::new(
            [0., 25.],
            Normalization::Linear,
            ColorMap::sequential(),
        )?);
        let mut a = Panel::new(
            Rect::new(21., 13., 72., 55.),
            Axes::new(
                [-2.5, nx as f64 * 5. - 2.5],
                [rmin, rmin + ny as f64 * dx],
                "Time (ms)",
                "Apparent range (km)",
            ),
        );
        a.push(Layer::Pcolormesh(Field::new(
            (0..=nx).map(|i| i as f64 * 5. - 2.5).collect(),
            (0..=ny).map(|i| rmin + i as f64 * dx).collect(),
            values,
            scale,
        )?));
        f.panels.push(a);
        f.colorbars.push(Colorbar::new(
            Rect::new(95., 13., 2.5, 55.),
            scale,
            "SNR (dB)",
            vec![0., 10., 20.],
        ));
        let mut b = Panel::new(
            Rect::new(130., 13., 55., 55.),
            Axes::new(
                [-2.5, nx as f64 * 5. - 2.5],
                [10., 26.],
                "Time (ms)",
                "Peak SNR (dB)",
            ),
        );
        b.push(Layer::Line(Line::series(
            t.iter().zip(&snr).map(|(x, y)| [*x, *y]).collect(),
            0,
        )));
        f.panels.push(b);
        let ms = utc[0] / 1_000_000;
        let day_ms = ms.rem_euclid(86400000);
        f.labels.push(Label::new(
            format!(
                "Event {}: {:02}:{:02}:{:06.3} UTC",
                event + 1,
                day_ms / 3600000,
                (day_ms / 60000) % 60,
                (day_ms % 60000) as f64 / 1000.
            ),
            95.,
            5.,
        ));
        f.save(out.join(path.file_stem().unwrap()).with_extension("png"))?;
    }
    Ok(())
}
