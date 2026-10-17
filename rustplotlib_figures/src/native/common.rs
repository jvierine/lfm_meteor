use anyhow::Result;
use rustplotlib::*;
pub const COLORS: [&str; 3] = ["#0072b2", "#d55e00", "#009e73"];
pub const SITES: [&str; 3] = ["Sanya", "Danzhou", "Wenchang"];
pub fn article_figure(mut source: Figure, fraction: f64) -> Result<Figure> {
    let article = ArticleLayout::new(184., 10.)?;
    let scale = 184. * fraction / source.width_mm;
    let mut figure = article.figure(fraction, source.height_mm * scale)?;
    let resize = |r: &mut Rect| {
        r.x *= scale;
        r.y *= scale;
        r.width *= scale;
        r.height *= scale;
    };
    for panel in &mut source.panels {
        resize(&mut panel.rect);
    }
    for bar in &mut source.colorbars {
        resize(&mut bar.rect);
    }
    for label in &mut source.labels {
        label.x *= scale;
        label.y *= scale;
        if let Some(point) = &mut label.leader {
            point[0] *= scale;
            point[1] *= scale;
        }
    }
    figure.panels = source.panels;
    figure.colorbars = source.colorbars;
    figure.labels = source.labels;
    figure.scales = source.scales;
    figure.check_label_overlap = source.check_label_overlap;
    figure.check_label_geometry = source.check_label_geometry;
    figure.auto_colorbar_layout = source.auto_colorbar_layout;
    article.check_inclusion(&figure, 184. * fraction)?;
    Ok(figure)
}
pub fn array(h: &hdf5::Group, name: &str) -> Result<Vec<f64>> {
    Ok(h.dataset(name)?.read_raw::<f64>()?)
}
pub fn quantile(values: &[f64], q: f64) -> f64 {
    let mut v: Vec<_> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let p = q * (v.len() - 1) as f64;
    let i = p.floor() as usize;
    v[i] + (v[p.ceil() as usize] - v[i]) * (p - i as f64)
}
pub fn limits(v: &[f64], pad: f64) -> [f64; 2] {
    let lo = v
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min);
    let hi = v
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let d = (hi - lo).max(lo.abs() * 1e-6).max(1e-6) * pad;
    [lo - d, hi + d]
}
pub fn points(x: &[f64], y: &[f64]) -> Vec<[f64; 2]> {
    x.iter().zip(y).map(|(&x, &y)| [x, y]).collect()
}
pub fn curve(p: &mut Panel, x: &[f64], y: &[f64], color: &str, width: f64) {
    let mut l = Line::new(points(x, y), color);
    l.width_pt = width;
    p.push(Layer::Line(l));
}
pub fn solid_scale(color: &str) -> Scale {
    let c = [0, 1, 2]
        .map(|i| u8::from_str_radix(&color[1 + 2 * i..3 + 2 * i], 16).unwrap() as f64 / 255.);
    Scale::new(
        [0., 1.],
        Normalization::Linear,
        ColorMap::new(vec![c, c]).unwrap(),
    )
    .unwrap()
}
pub fn dots(
    p: &mut Panel,
    x: &[f64],
    y: &[f64],
    color: &str,
    size: f64,
    opacity: f64,
    hollow: bool,
) -> Result<()> {
    let scale = solid_scale(if hollow { "#ffffff" } else { color });
    p.scatter_colored(
        &points(x, y),
        &vec![0.5; x.len()],
        &scale,
        size,
        opacity,
        color,
    )
}
pub fn rect(p: &mut Panel, x: [f64; 2], y: [f64; 2], color: &str, opacity: f64) {
    let mut l = Line::new(
        vec![
            [x[0], y[0]],
            [x[1], y[0]],
            [x[1], y[1]],
            [x[0], y[1]],
            [x[0], y[0]],
        ],
        color,
    );
    l.width_pt = 0.5;
    l.fill = Some(color.into());
    l.fill_opacity = opacity;
    p.push(Layer::Line(l));
}
pub fn panel_label(f: &mut Figure, p: &Panel, text: &str) {
    let mut l = Label::new(format!("\\textbf{{{text}}}"), p.rect.x + 3., p.rect.y + 3.);
    l.background = Some("white".into());
    l.padding_mm = 0.4;
    f.labels.push(l);
}
pub fn histogram(v: &[f64], edges: &[f64]) -> Vec<f64> {
    let mut count = vec![0.; edges.len() - 1];
    for &x in v {
        if x.is_finite() && x >= edges[0] && x <= *edges.last().unwrap() {
            let i = edges
                .partition_point(|&e| e <= x)
                .saturating_sub(1)
                .min(count.len() - 1);
            count[i] += 1.;
        }
    }
    count
}
