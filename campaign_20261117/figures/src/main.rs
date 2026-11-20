mod common;
mod detection;
mod height;
mod mass;
mod rcs;
mod radiants;
mod orbits;
mod review_event;
use anyhow::{bail, Result};
use rustplotlib::*;
use std::path::Path;

fn residual(h: &hdf5::File) -> Result<Figure> {
    use common::*;
    let values = array(h, "residual_m")?;
    let edges: Vec<_> = (0..=200).map(|i| -50. + i as f64 * 0.5).collect();
    let counts = histogram(&values, &edges);
    let article = ArticleLayout::new(184., 10.)?;
    let mut f = article.figure(0.5, 61.)?;
    let high = counts.iter().copied().fold(1_f64, f64::max)*1.25;
    let mut a = Axes::new([-50.,50.], [1.,high], "Path residual (m)", "Count");
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
        name => bail!("unknown figure {name}"),
    };
    let path = Path::new(&args[3]);
    std::fs::create_dir_all(path.parent().unwrap())?;
    fig.latex.cache=path.parent().unwrap().join(".latex-cache");
    for extension in ["pdf","png"] { fig.save(path.with_extension(extension))?; }
    Ok(())
}
