use anyhow::Result;
use rustplotlib::*;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len() == 5, "beam_check APRIL NOV17 NOV21 OUTPUT");
    let files = args[1..4].iter().map(hdf5::File::open).collect::<hdf5::Result<Vec<_>>>()?;
    let read = |h: &hdf5::File, key: &str| h.dataset(key)?.read_raw::<f64>();
    let mut max = 10_f64;
    for h in &files {
        for v in read(h, "counts")? {
            if v.is_finite() { max = max.max(v); }
        }
    }
    let mut f = ArticleLayout::new(232., 10.)?.figure(1., 82.)?;
    let id = f.scale(Scale::new([1., max], Normalization::Log, ColorMap::inferno())?);
    for (i, (h, label)) in files.iter().zip(["22 April 2024", "17 November 2026", "21 November 2026"]).enumerate() {
        let x = 17. + 68. * i as f64;
        f.labels.push(Label::new(label, x + 28., 5.));
        let mut a = Axes::new([-2.6, 2.6], [-2.6, 2.6], "East offset (deg)",
            if i == 0 { "North offset (deg)" } else { "" });
        a.xticks = vec![-2., 0., 2.];
        a.yticks = vec![-2., 0., 2.];
        let mut p = Panel::new(Rect::new(x, 11., 56., 56.), a);
        let mut counts = read(h, "counts")?;
        for v in &mut counts {
            if *v < 1. { *v = f64::NAN; }
        }
        p.push(Layer::Pcolormesh(Field::new(read(h, "east_edges")?, read(h, "north_edges")?, counts, id)?));
        let g = read(h, "gain_grid_deg")?;
        let model = read(h, "gain_430_check_db")?;
        for level in [-30., -20., -13., -3.] {
            let c = Contour::new(g.clone(), g.clone(), model.clone(), vec![level,level+1e-9], id, false)?;
            let pts: Vec<_> = c.segments(level).into_iter().flat_map(|s| [s[0],s[1],[f64::NAN;2]]).collect();
            if level == -3. {
                let mut l = Line::new(pts.clone(), "#ffffff");l.width_pt = 2.;p.push(Layer::Line(l));
            }
            let mut l = Line::new(pts, if level == -3. { "#161616" } else { "#777777" });
            l.width_pt = if level == -3. { 1. } else { 0.5 };
            p.push(Layer::Line(l));
        }
        f.panels.push(p);
    }
    let ticks = [1.,10.,100.].into_iter().filter(|&v| v <= max).collect();
    f.colorbars.push(Colorbar::new(Rect::new(213.,11.,2.,56.), id, "Count", ticks));
    f.save(format!("{}.pdf",args[4]))?;
    f.save(format!("{}.png",args[4]))?;
    Ok(())
}
