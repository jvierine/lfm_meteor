use anyhow::Result;
use rustplotlib::*;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len() == 3, "campaign_heights INPUT.h5 OUTPUT.pdf");
    let h = hdf5::File::open(&args[1])?;
    let read = |key: &str| h.dataset(key)?.read_raw::<f64>();
    let bins = read("height_edges")?;
    let series = read("height_counts")?;
    let mono = read("monostatic_counts")?;
    let max = series.iter().copied().fold(1., f64::max);
    let mono_max = mono.iter().copied().fold(1., f64::max);
    let mut f = ArticleLayout::new(178.54, 9.)?.figure(1., 90.)?;
    let mut axes = Axes::new([0.,1.08*max],[80.,120.],"Tri-static events per km","Height (km)");
    let mono_scale = max/mono_max;
    let tick = if mono_max >= 2000. { 500. } else { 250. };
    axes.secondary_x = Some(SecondaryAxis::new("Sanya monostatic events per km",
        (0..=(1.08*mono_max/tick) as usize).map(|i| tick*i as f64).collect(),
        |v| v*mono_scale)?);
    let mut p = Panel::new(Rect::new(17., 23., 65., 53.),axes);
    p.major_grid("#e4e4e4", 0.4);
    let mut pts = vec![[0.,bins[0]]];
    for (j,n) in mono.iter().enumerate() {
        pts.extend([[n*mono_scale,bins[j]],[n*mono_scale,bins[j+1]]]);
    }
    pts.push([0.,*bins.last().unwrap()]);
    let mut line = Line::new(pts, "#111111");
    line.width_pt = 1.15;
    line.dash = Some("3 2".into());
    let mut halo = Line::new(line.points.clone(), "#ffffff");
    halo.width_pt = 2.1;
    halo.dash = line.dash.clone();
    let labels = ["22 April 2024", "17 November 2026", "21 November 2026"];
    let colors = ["#0072b2","#d55e00","#009e73"];
    for (i, v) in series.chunks_exact(bins.len()-1).enumerate() {
        let mut pts = vec![[0.,bins[0]]];
        for (j, n) in v.iter().enumerate() {
            pts.extend([[*n,bins[j]],[*n,bins[j+1]]]);
        }
        pts.push([0.,*bins.last().unwrap()]);
        let mut line = Line::new(pts, colors[i]);
        line.width_pt = 1.25;
        p.push(Layer::Line(line));
    }
    p.push(Layer::Line(halo));
    p.push(Layer::Line(line));
    let mut entries: Vec<_> = labels.into_iter().zip(colors)
        .map(|(date, color)| LegendEntry::line(date, color)).collect();
    entries.push(LegendEntry {
        label: "All monostatic".into(),
        symbol: LegendSymbol::Line {
            color: "#111111".into(), width_pt: 1.15, dash: Some("3 2".into()),
        },
    });
    let mut legend = Legend::new(entries);
    legend.columns = 4;
    legend.position = LegendPosition::AboveCenter;
    legend.padding_mm = 1.;
    let mut legend_panel = Panel::new(
        Rect::new(2., 1., 174.54, 12.),
        Axes::new([0., 1.], [0., 1.], "", ""),
    );
    legend_panel.axes_visible = false;
    legend_panel.legend = Some(legend);
    f.panels.push(legend_panel);
    f.panels.push(p);
    let counts = read("counts")?;
    let high = counts.iter().filter(|v| v.is_finite()).copied().fold(1.,f64::max);
    let id = f.scale(Scale::new([1.,high],Normalization::Log,ColorMap::inferno())?);
    let mut a = Axes::new([-2.6,2.6],[-2.6,2.6],"East offset (deg)","North offset (deg)");
    a.xticks = vec![-2.,0.,2.]; a.yticks = vec![-2.,0.,2.];
    let mut p = Panel::new(Rect::new(101.,23.,53.,53.),a);
    p.push(Layer::Pcolormesh(Field::new(read("edges")?,read("edges")?,counts,id)?));
    let g = read("gain_grid")?;
    let gain = read("gain_db")?;
    for lv in [-30.,-20.,-13.,-3.] {
        let c = Contour::new(g.clone(),g.clone(),gain.clone(),vec![lv,lv+1e-9],id,false)?;
        let pts: Vec<_> = c.segments(lv).into_iter().flat_map(|s|[s[0],s[1],[f64::NAN;2]]).collect();
        if lv == -3. {
            let mut halo = Line::new(pts.clone(),"#ffffff");
            halo.width_pt = 2.;
            p.push(Layer::Line(halo));
        }
        let mut line = Line::new(pts,if lv == -3. {"#161616"} else {"#777777"});
        line.width_pt = if lv == -3. {1.} else {0.55};
        p.push(Layer::Line(line));
    }
    f.panels.push(p);
    f.labels.push(Label::new(r"\textbf{(a)}",11.,19.));
    f.labels.push(Label::new(r"\textbf{(b)}",96.,19.));
    f.colorbars.push(Colorbar::new(Rect::new(158.,23.,2.,53.),id,"Count",
        [1.,10.,100.].into_iter().filter(|x| *x <= high).collect()));
    f.save(&args[2])?;
    Ok(())
}
