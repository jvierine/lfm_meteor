use anyhow::Result;
use rustplotlib::*;
use std::path::PathBuf;

fn main() -> Result<()> {
    let folder = PathBuf::from(std::env::args().nth(1).expect("event folder"));
    let h = hdf5::File::open(folder.join("plot.h5"))?;
    let model = h.dataset("modeled_path_m")?.read_raw::<f64>()?;
    let observed = h.dataset("seed_apparent_path_m")?.read_raw::<f64>()?;
    let station = h.dataset("station")?.read_raw::<i64>()?;
    // Match the residual sign used by voltage-event.
    let residual: Vec<f64> = model.iter().zip(&observed).map(|(a,b)| a-b).collect();
    let width = 2.;
    let limit = (residual.iter().copied().map(f64::abs).fold(0., f64::max)/width).ceil()*width + width;
    let nbins = (2.*limit/width) as usize;
    let edges: Vec<f64> = (0..=nbins).map(|i| -limit+i as f64*width).collect();
    let colors = ["#0072B2", "#D55E00", "#009E73"];
    let names = ["Sanya", "Danzhou", "Wenchang"];
    let mut counts = vec![vec![0.;nbins];3];
    for (&r,&s) in residual.iter().zip(&station) {
        counts[s as usize][((r+limit)/width).floor() as usize] += 1.;
    }
    let ymax=counts.iter().flatten().copied().fold(0., f64::max)+1.;
    let catalogue=residual.len()>100;
    let baseline=if catalogue {0.7} else {0.};
    let mut f=Figure::new(IntendedUse::ArticleWide);
    f.width_mm=180.; f.height_mm=104.;
    let mut axes=Axes::new([-limit,limit],[baseline,ymax*1.1],"Apparent path residual (m)","Measurements per 2 m bin");
    if catalogue {
        axes.set_yscale(AxisScale::Log10)?;
        axes.yticks=(0..=5).map(|x|10_f64.powi(x)).filter(|x|*x<=ymax).collect();
    } else {axes.yticks=(0..=ymax as usize).map(|x|x as f64).collect();}
    let mut p=Panel::new(Rect::new(22.,27.,150.,58.),axes);
    let mut entries=Vec::new();
    let out=hdf5::File::create(folder.join("delay_residual_histogram.h5"))?;
    out.new_dataset_builder().with_data(&residual).create("model_minus_observed_m")?;
    out.new_dataset_builder().with_data(&station).create("station")?;
    out.new_dataset_builder().with_data(&edges).create("bin_edges_m")?;
    for s in 0..3 {
        let mut xy=vec![[edges[0],baseline]];
        for i in 0..nbins {let y=counts[s][i].max(baseline);xy.extend([[edges[i],y],[edges[i+1],y]]);}
        xy.push([edges[nbins],baseline]);
        p.push(Layer::Line(Line::new(xy,colors[s])));
        let n=station.iter().filter(|&&x|x==s as i64).count();
        let rms=(residual.iter().zip(&station).filter(|(_,j)|**j==s as i64).map(|(r,_)|r*r).sum::<f64>()/n as f64).sqrt();
        entries.push(LegendEntry::line(format!("{}: $n={}$, RMS ${:.1}$ m",names[s],n,rms),colors[s]));
        out.new_dataset_builder().with_data(&counts[s]).create(names[s])?;
        println!("{} n={} RMS={:.4} m",names[s],n,rms);
    }
    let mut legend=Legend::new(entries);legend.position=LegendPosition::AboveCenter;legend.columns=1;
    p.legend=Some(legend);f.panels.push(p);
    f.latex.cache=folder.parent().unwrap().join(".rustplotlib-cache");
    for ext in ["png","pdf"] {f.save(folder.join(format!("delay_residual_histogram.{ext}")))?;}
    Ok(())
}
