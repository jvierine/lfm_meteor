use super::common::*;
use anyhow::Result;
use rustplotlib::*;
pub fn figure(h: &hdf5::File) -> Result<Figure> {
    let tri = array(h, "tristatic_height_km")?;
    let mono = array(h, "monostatic_height_km")?;
    let edges: Vec<_> = (80..=120).map(|i| i as f64).collect();
    let tc = histogram(&tri, &edges);
    let mc = histogram(&mono, &edges);
    let mut f = Figure::new(IntendedUse::Custom {
        width_mm: 240.,
        font_pt: 9.5,
    });
    f.height_mm = 91.;
    let mut a = Axes::new(
        [0., mc.iter().copied().fold(0_f64, f64::max) * 1.08],
        [80., 120.],
        "Count per km",
        "Height (km)",
    );
    a.xticks = vec![0., 2000., 4000.];
    a.yticks = vec![80., 90., 100., 110., 120.];
    let mut p = Panel::new(Rect::new(23., 15., 54., 59.), a);
    p.major_grid("#e4e4e4", 0.4);
    let mut step = vec![[0., 80.]];
    for i in 0..tc.len() {
        rect(
            &mut p,
            [0., tc[i]],
            [edges[i] + 0.1, edges[i + 1] - 0.1],
            "#7b3294",
            0.95,
        );
        step.extend([[mc[i], edges[i]], [mc[i], edges[i + 1]]]);
    }
    p.push(Layer::Line(Line::new(step, COLORS[0])));
    let mut l = Legend::new(vec![
        LegendEntry::line("Monostatic", COLORS[0]),
        LegendEntry::line("Tri-static", "#7b3294"),
    ]);
    l.position = LegendPosition::AboveCenter;
    l.padding_mm = 0.;
    l.avoid_data = false;
    p.legend = Some(l);
    panel_label(&mut f, &p, "(a)");
    f.panels.push(p);
    let mut a = Axes::new([-5., 43.], [80., 120.], "Receiver gain (dBi)", "");
    a.show_y_axis = false;
    a.xticks = vec![0., 20., 40.];
    a.yticks = vec![80., 90., 100., 110., 120.];
    let mut p = Panel::new(Rect::new(80., 15., 51., 59.), a);
    p.major_grid("#e4e4e4", 0.4);
    let height = array(h, "gain_height_km")?;
    for (site, color) in [("Danzhou", COLORS[1]), ("Wenchang", COLORS[2])] {
        curve(
            &mut p,
            &array(h, &format!("gain_{site}"))?,
            &height,
            color,
            1.4,
        );
    }
    let mut l = Legend::new(vec![
        LegendEntry::line("Danzhou", COLORS[1]),
        LegendEntry::line("Wenchang", COLORS[2]),
    ]);
    l.columns = 1;
    l.position = LegendPosition::AboveCenter;
    l.avoid_data = false;
    l.padding_mm = 0.;
    p.legend = Some(l);
    panel_label(&mut f, &p, "(b)");
    f.panels.push(p);
    let counts = array(h, "counts")?;
    let xe = array(h, "east_edges")?;
    let ye = array(h, "north_edges")?;
    let max = counts
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(1_f64, f64::max);
    let cid = f.scale(Scale::new(
        [1., max],
        Normalization::Log,
        ColorMap::inferno(),
    )?);
    let mut a = Axes::new(
        [-2.6, 2.6],
        [-2.6, 2.6],
        "East offset (deg)",
        "North offset (deg)",
    );
    a.xticks = vec![-2., 0., 2.];
    a.yticks = vec![-2., 0., 2.];
    let mut p = Panel::new(Rect::new(156., 15., 59., 59.), a);
    p.push(Layer::Pcolormesh(Field::new(xe, ye, counts, cid)?));
    let grid = array(h, "gain_grid_deg")?;
    let gain = array(h, "gain_db")?;
    let level = h.attr("containment_level_db")?.read_scalar::<f64>()?;
    for (lv, color, width) in [
        (-30., "#777777", 0.5),
        (-20., "#666666", 0.5),
        (-13., "#444444", 0.6),
        (-3., "#222222", 1.),
        (level, "#25734a", 0.8),
    ] {
        let c = Contour::new(
            grid.clone(),
            grid.clone(),
            gain.clone(),
            vec![lv, lv + 1e-9],
            cid,
            false,
        )?;
        let mut points = Vec::new();
        for s in c.segments(lv) {
            points.extend([s[0], s[1], [f64::NAN; 2]]);
        }
        let mut line = Line::new(points, color);
        line.width_pt = width;
        p.push(Layer::Line(line));
    }
    panel_label(&mut f, &p, "(c)");
    f.panels.push(p);
    f.colorbars.push(Colorbar::new(
        Rect::new(220., 15., 2., 59.),
        cid,
        "Count",
        vec![1., 10.],
    ));
    article_figure(f, 1.)
}
