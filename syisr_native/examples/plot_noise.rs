use anyhow::{ensure, Result};
use rustplotlib::*;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 3, "usage: plot_noise INPUT.h5 OUTPUT.png");
    let h = hdf5::File::open(&args[1])?;
    let t = h.dataset("averages/utc_ns")?.read_raw::<i64>()?;
    let y = h.dataset("averages/raw_noise_power_adc2")?.read_raw::<f64>()?;
    ensure!(!t.is_empty() && t.len() == y.len(), "invalid noise series");
    let day = t[0].div_euclid(86_400_000_000_000) * 86_400_000_000_000;
    let points: Vec<_> = t.iter().zip(&y).filter(|(_, v)| v.is_finite() && **v > 0.)
        .map(|(t, y)| [(*t - day) as f64 / 3.6e12, *y / 1e6]).collect();
    ensure!(!points.is_empty(), "no positive finite noise estimates");
    let lo = points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    let hi = points.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
    let pad = ((hi - lo) * 0.05).max(hi * 0.001);
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 170.;
    f.height_mm = 85.;
    let date = chrono::DateTime::from_timestamp_nanos(day).format("%Y-%m-%d");
    let mut p = Panel::new(Rect::new(28., 8., 135., 59.),
        Axes::new([points[0][0], points.last().unwrap()[0]],
            [lo - pad, hi + pad], format!("UTC hour ({date})"),
            r"Noise power ($10^6$ ADC counts$^2$)"));
    let mut line = Line::new(points, "#0072B2");
    line.width_pt = 0.6;
    p.push(Layer::Line(line));
    f.panels.push(p);
    f.save(&args[2])?;
    Ok(())
}
