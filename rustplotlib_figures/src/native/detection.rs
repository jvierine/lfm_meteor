use super::common::*;
use anyhow::Result;
use hdf5::types::VarLenUnicode;
use rustplotlib::*;
fn clock(hour: f64) -> String {
    let minutes = (hour * 60.).round() as i64;
    format!(
        "{:02}:{:02}",
        (minutes / 60).rem_euclid(24),
        minutes.rem_euclid(60)
    )
}
pub fn figure(h: &hdf5::File) -> Result<Figure> {
    let observations = h.group("observations")?;
    let names = observations.dataset("site")?.read_raw::<VarLenUnicode>()?;
    let time = observations.dataset("time_ns")?.read_raw::<i64>()?;
    let delay = array(&observations, "delay_us")?;
    let day = h.attr("utc_day_start_ns")?.read_scalar::<i64>()?;
    let hours: Vec<_> = time.iter().map(|&v| (v - day) as f64 / 3.6e12).collect();
    let xr = [12., 24.];
    let xticks = vec![12., 14., 16., 18., 20., 22., 24.];
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.height_mm = 154.;
    f.font_pt = 9.;
    f.latex.font_pt = 9.;
    let mut a = Axes::new(xr, limits(&delay, 0.03), "", r"Delay ($\mu\mathrm{s}$)");
    a.show_x_axis = false;
    a.xticks = xticks.clone();
    let offset = 109.6222 / 15.;
    let solar: Vec<_> = xticks.iter().map(|v| (v + offset) % 24.).collect();
    a.secondary_x = Some(SecondaryAxis {
        label: "Local solar time at Sanya (sunrise = 06:00)".into(),
        ticks: xticks.iter().zip(&solar).map(|(&x, &s)| (x, s)).collect(),
        tick_labels: Some(solar.iter().map(|&v| clock(v)).collect()),
    });
    let mut p = Panel::new(Rect::new(22., 20., 147., 37.), a);
    p.major_grid("#e4e4e4", 0.4);
    for col in 0..3 {
        let ids: Vec<_> = (0..names.len())
            .filter(|&i| names[i].as_str().eq_ignore_ascii_case(SITES[col]))
            .collect();
        let x: Vec<_> = ids.iter().map(|&i| hours[i]).collect();
        let y: Vec<_> = ids.iter().map(|&i| delay[i]).collect();
        dots(&mut p, &x, &y, COLORS[col], 1.8, 0.62, false)?;
    }
    // Dense detection markers are still native Rustplotlib geometry; rasterize
    // only this native marker layer to keep the scientific PDF reasonably small.
    let count = p.layers.len();
    p.rasterize_lines(1..count, 220.)?;
    f.panels.push(p);
    let counts = h.group("counts")?;
    let mut max = 0_f64;
    for name in ["sanya", "danzhou", "wenchang", "tristatic"] {
        max = max.max(
            array(&counts.group(name)?, "count")?
                .iter()
                .copied()
                .fold(0_f64, f64::max),
        );
    }
    let mut a = Axes::new(xr, [1., max * 1.25], "", "Events per hour");
    a.show_x_axis = false;
    a.xticks = xticks.clone();
    a.set_yscale(AxisScale::Log10)?;
    let mut p = Panel::new(Rect::new(22., 66., 147., 30.), a);
    p.major_grid("#e4e4e4", 0.4);
    for (col, name) in ["sanya", "danzhou", "wenchang", "tristatic"]
        .iter()
        .enumerate()
    {
        let g = counts.group(name)?;
        let hour = array(&g, "hour")?;
        let n = array(&g, "count")?;
        let mut pts = Vec::new();
        for (&x, &v) in hour.iter().zip(&n) {
            if v > 0. {
                pts.extend([[x - 0.5, v], [x + 0.5, v]]);
            } else {
                pts.push([f64::NAN; 2]);
            }
        }
        let mut l = Line::new(pts, if col < 3 { COLORS[col] } else { "#222222" });
        l.width_pt = 1.2;
        if col == 3 {
            l.dash = Some("2 1".into());
        }
        p.push(Layer::Line(l));
    }
    let mut legend = Legend::new(vec![LegendEntry {
        label: "Tri-static candidates".into(),
        symbol: LegendSymbol::Line {
            color: "#222222".into(),
            width_pt: 1.2,
            dash: Some("2 1".into()),
        },
    }]);
    legend.position = LegendPosition::AboveCenter;
    legend.padding_mm = 0.5;
    legend.avoid_data = false;
    p.legend = Some(legend);
    f.panels.push(p);
    let noise = h.group("noise")?;
    let mut ymax = 0_f64;
    for site in ["sanya", "danzhou", "wenchang"] {
        let g = noise.group(site)?;
        for key in ["floor_tsys_k", "model_tsys_k"] {
            ymax = ymax.max(
                array(&g, key)?
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .fold(0_f64, f64::max),
            );
        }
    }
    let mut a = Axes::new(xr, [0., ymax * 1.04], "UTC time", "Noise temperature (K)");
    a.xticks = xticks;
    a.xticklabels = Some(a.xticks.iter().map(|&v| clock(v)).collect());
    a.yticks = (0..=(ymax * 1.04 / 200.) as usize)
        .map(|i| i as f64 * 200.)
        .collect();
    let mut p = Panel::new(Rect::new(22., 105., 147., 35.), a);
    p.major_grid("#e4e4e4", 0.4);
    for (col, site) in ["sanya", "danzhou", "wenchang"].iter().enumerate() {
        let g = noise.group(site)?;
        let x = array(&g, "floor_hour")?;
        let y = array(&g, "floor_tsys_k")?;
        dots(&mut p, &x, &y, COLORS[col], 2., 0.6, false)?;
        curve(
            &mut p,
            &array(&g, "model_hour")?,
            &array(&g, "model_tsys_k")?,
            COLORS[col],
            1.2,
        );
        let rec = g.attr("receiver_temperature_k")?.read_scalar::<f64>()?;
        let mut l = Line::new(vec![[12., rec], [24., rec]], COLORS[col]);
        l.opacity = 0.4;
        l.dash = Some("0.5 1.2".into());
        l.width_pt = 0.8;
        p.push(Layer::Line(l));
    }
    let mut legend = Legend::new(
        (0..3)
            .map(|i| LegendEntry::circle(SITES[i], COLORS[i]))
            .collect(),
    );
    legend.columns = 3;
    legend.position = LegendPosition::UpperLeft;
    legend.avoid_data = false;
    legend.padding_mm = 0.5;
    p.legend = Some(legend);
    f.panels.push(p);
    article_figure(f, 1.)
}
