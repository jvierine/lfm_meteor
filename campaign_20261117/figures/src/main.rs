mod common;
mod detection;
mod height;
mod mass;
mod rcs;
mod radiants;
mod orbits;
mod review_event;
mod pointing;
use anyhow::{bail, Result};
use rustplotlib::*;
use std::path::Path;

fn residual(h: &hdf5::File) -> Result<Figure> {
    use common::*;
    if h.link_exists("station") {
        let values = array(h, "residual_m")?;
        let station = h.dataset("station")?.read_raw::<i64>()?;
        anyhow::ensure!(values.len() == station.len() && values.iter().all(|r| r.abs() < 40.));
        let edges: Vec<_> = (-80..=80).map(|i| i as f64 * 0.5).collect();
        let counts: Vec<_> = (0..3).map(|i| {
            let v: Vec<_> = values.iter().zip(&station).filter_map(|(&r,&s)| (i==s).then_some(r)).collect();
            histogram(&v, &edges)
        }).collect();
        anyhow::ensure!(counts.iter().flatten().sum::<f64>() == values.len() as f64);
        let mut f = ArticleLayout::new(92., 9.)?.figure(1., 49.)?;
        let high = counts.iter().flatten().copied().fold(1.,f64::max)*1.2;
        let mut a = Axes::new([-40.,40.], [1.,high], "Path residual (m)", "Count");
        a.xticks = vec![-40.,-20.,0.,20.,40.];
        a.set_yscale(AxisScale::Log10)?;
        let mut p = Panel::new(Rect::new(16.,9.,72.,29.),a);
        let colors = ["#0072b2","#d55e00","#009e73"];
        for (i,c) in counts.iter().enumerate() {
            let mut steps = Vec::new();
            for (j,&n) in c.iter().enumerate() {
                if n > 0. { steps.extend([[edges[j],n],[edges[j+1],n]]); }
                else { steps.push([f64::NAN;2]); }
            }
            p.push(Layer::Line(Line::new(steps, colors[i])));
        }
        let mut key = Panel::new(Rect::new(13.,1.,75.,7.), Axes::new([13.,88.],[0.,1.],"",""));
        key.axes_visible = false;
        for (i,(name,color)) in ["Sanya","Danzhou","Wenchang"].into_iter().zip(colors).enumerate() {
            let x = 15. + 24.5*i as f64;
            key.push(Layer::Line(Line::new(vec![[x,0.5],[x+5.,0.5]],color)));
            f.labels.push(Label::new(name,x+12.5,4.5));
        }
        f.panels.push(key);
        f.panels.push(p);
        return Ok(f);
    }
    let values: Vec<_> = array(h, "residual_m")?.into_iter().map(|v| v.clamp(-50., 50.)).collect();
    let edges: Vec<_> = (-101..=101).map(|i| i as f64 * 0.5).collect();
    let counts = histogram(&values, &edges);
    anyhow::ensure!(counts.iter().sum::<f64>() == values.len() as f64, "residual histogram lost samples");
    let article = ArticleLayout::new(184., 10.)?;
    let mut f = article.figure(0.5, 61.)?;
    let high = counts.iter().copied().fold(1_f64, f64::max)*1.25;
    let mut a = Axes::new([edges[0], edges[edges.len()-1]], [1.,high], "Path residual (m)", "Count");
    a.xticks = vec![-50., -25., 0., 25., 50.];
    a.set_yscale(AxisScale::Log10)?;
    let mut p = Panel::new(Rect::new(17.,5.,70.,42.), a);
    let mut pts = Vec::new();
    for i in 0..counts.len() {
        let y = counts[i];
        if y > 0. { pts.extend([[edges[i],y],[edges[i+1],y]]); }
        else { pts.push([f64::NAN;2]); }
    }
    p.push(Layer::Line(Line::new(pts, "#0072b2")));
    f.panels.push(p);
    Ok(f)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len()==4, "usage: november-memo-figures KIND INPUT.h5 OUTPUT_BASE");
    let h = hdf5::File::open(&args[2])?;
    let mut fig = match args[1].as_str() {
        "figure2" => detection::figure(&h)?,
        "figure3" => height::figure(&h)?,
        "figure4" => residual(&h)?,
        "event" => review_event::figure(&h)?,
        "figure7" => mass::figure(&h)?,
        "figure8" => rcs::figure(&h)?,
        "figure9" => radiants::figure(&h)?,
        "figure10" => orbits::figure(&h)?,
        "pointing" => pointing::figure(&h)?,
        name => bail!("unknown figure {name}"),
    };
    let path = Path::new(&args[3]);
    std::fs::create_dir_all(path.parent().unwrap())?;
    fig.latex.cache=path.parent().unwrap().join(".latex-cache");
    for extension in ["pdf","png"] { fig.save(path.with_extension(extension))?; }
    Ok(())
}
