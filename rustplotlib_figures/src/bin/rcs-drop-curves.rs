use anyhow::Result;
use hdf5::types::VarLenUnicode;
use rustplotlib::*;
use std::path::{Path, PathBuf};

fn dots(p: &mut Panel, points: &[[f64; 2]], color: &str, size: f64) -> Result<()> {
    let c = [0, 1, 2].map(|i| u8::from_str_radix(&color[1+2*i..3+2*i], 16).unwrap() as f64 / 255.);
    let scale = Scale::new([0., 1.], Normalization::Linear, ColorMap::new(vec![c, c])?)?;
    p.scatter_colored(points, &vec![0.5; points.len()], &scale, size, 1., color)
}

fn page(h: &hdf5::File, ids: &[String], scores: &[f64], indices: &[usize], output: &Path, drop: bool) -> Result<()> {
    let mut f = Figure::new(IntendedUse::Custom { width_mm: 220., font_pt: 8.5 });
    let rows = indices.len().div_ceil(3);
    f.height_mm = 12. + rows as f64 * 65.;
    f.raster_dpi = 170.;
    f.latex.cache = output.parent().unwrap().join(".rustplotlib-cache");
    let window = h.attr("primary_window_ms")?.read_scalar::<f64>()?;
    let abrupt = h.attr("detector").is_ok();
    let case = h.group(&format!("cases/snr20_gain10_window{window:.0}"))?;
    let high = case.dataset("high_index")?.read_raw::<i64>()?;
    let low = case.dataset("low_index")?.read_raw::<i64>()?;
    for (slot, &index) in indices.iter().enumerate() {
        let g = h.group(&format!("events/{}", ids[index]))?;
        let read = |key: &str| -> Result<Vec<f64>> { Ok(g.dataset(key)?.read_raw::<f64>()?) };
        let time = read("time_ms")?;
        let rcs = read("rcs_dbsm")?;
        let snr = read("snr_db")?;
        let gain = read("relative_gain_db")?;
        let retained = g.dataset("retained")?.read_raw::<bool>()?;
        let finite: Vec<_> = rcs.iter().copied().filter(|x| x.is_finite()).collect();
        let ymin = (finite.iter().copied().fold(f64::INFINITY, f64::min) / 5.).floor() * 5. - 2.;
        let ymax = (finite.iter().copied().fold(f64::NEG_INFINITY, f64::max) / 5.).ceil() * 5. + 2.;
        let end = *time.last().unwrap();
        let rect = Rect::new(20. + (slot % 3) as f64 * 72., 12. + (slot / 3) as f64 * 65., 53., 43.);
        let mut panel = Panel::new(rect, Axes::new([-0.02 * end, 1.02 * end], [ymin, ymax], "Time (ms)", "RCS (dBsm)"));
        let full: Vec<_> = time.iter().zip(&rcs).map(|(&t, &r)| [t, r]).collect();
        let mut line = Line::new(full.clone(), "#b8b8b8"); line.width_pt = 0.8;
        panel.push(Layer::Line(line));
        dots(&mut panel, &full.iter().copied().filter(|p| p[1].is_finite()).collect::<Vec<_>>(), "#b8b8b8", 2.)?;
        let color = if drop { "#c0392b" } else { "#0072b2" };
        let good: Vec<_> = (0..time.len()).filter(|&k| retained[k] && rcs[k].is_finite() && snr[k] >= 20. && gain[k] >= -10.).map(|k| full[k]).collect();
        dots(&mut panel, &good, color, 2.8)?;
        if high[index] >= 0 && low[index] >= 0 {
            let i = high[index] as usize; let j = low[index] as usize;
            let median = |v: &[f64]| { let mut a = v.to_vec(); a.sort_by(f64::total_cmp); a[a.len()/2] };
            let top = if abrupt { full[i] } else { [median(&time[i..i+3]), median(&rcs[i..i+3])] };
            let bottom = if abrupt { full[j] } else { [median(&time[j..j+3]), median(&rcs[j..j+3])] };
            let mut guide = Line::new(vec![top, bottom], "#171717");
            guide.dash = Some("2 2".into()); guide.width_pt = 0.8;
            panel.push(Layer::Line(guide));
            let markers = if abrupt { vec![top, bottom, full[j+1]] } else { vec![top, bottom] };
            dots(&mut panel, &markers, "#171717", 3.5)?;
        }
        let number = ids[index].split('_').nth(1).unwrap();
        f.labels.push(Label::new(format!("{}: {:.1} dB", number, scores[index].max(0.)), rect.x + rect.width / 2., rect.y - 3.));
        f.panels.push(panel);
    }
    for ext in ["png", "pdf"] { f.save(output.with_extension(ext))?; }
    Ok(())
}

fn main() -> Result<()> {
    let path = PathBuf::from(std::env::args().nth(1).expect("survey HDF5"));
    let h = hdf5::File::open(&path)?;
    let ids: Vec<String> = h.dataset("event_id")?.read_raw::<VarLenUnicode>()?.iter().map(ToString::to_string).collect();
    let window = h.attr("primary_window_ms")?.read_scalar::<f64>()?;
    let g = h.group(&format!("cases/snr20_gain10_window{window:.0}"))?;
    let scores = g.dataset("drop_db")?.read_raw::<f64>()?;
    let eligible = g.dataset("eligible")?.read_raw::<bool>()?;
    let threshold = h.attr("primary_threshold_db")?.read_scalar::<f64>()?;
    let out = path.parent().unwrap();
    for (name, is_drop) in [("drop", true), ("below_threshold", false)] {
        let mut indices: Vec<_> = (0..ids.len()).filter(|&i| eligible[i] && (scores[i] >= threshold) == is_drop).collect();
        indices.sort_by(|&i, &j| scores[j].total_cmp(&scores[i]));
        println!("{name}: {} events", indices.len());
        for (n, chunk) in indices.chunks(9).enumerate() {
            page(&h, &ids, &scores, chunk, &out.join(format!("{name}_{:02}", n+1)), is_drop)?;
        }
        if !is_drop {
            let typical: Vec<_> = (0..9).map(|k| indices[k * (indices.len()-1) / 8]).collect();
            page(&h, &ids, &scores, &typical, &out.join("no_drop_examples"), false)?;
        }
    }
    Ok(())
}
