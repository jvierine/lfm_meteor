use anyhow::Result;
use rustplotlib::*;
use std::path::PathBuf;

fn median(values: &[f64]) -> f64 {
    let mut sorted=values.to_vec();sorted.sort_by(f64::total_cmp);
    let n=sorted.len();
    if n%2==0 {(sorted[n/2-1]+sorted[n/2])/2.} else {sorted[n/2]}
}

fn main() -> Result<()> {
    let folder = PathBuf::from(std::env::args().nth(1).expect("event folder"));
    let h = hdf5::File::open(folder.join("plot.h5"))?;
    let model = h.dataset("modeled_path_m")?.read_raw::<f64>()?;
    let observed = h.dataset("seed_apparent_path_m")?.read_raw::<f64>()?;
    let station = h.dataset("station")?.read_raw::<i64>()?;
    // Match the residual sign used by voltage-event.
    let residual: Vec<f64> = model.iter().zip(&observed).map(|(a,b)| a-b).collect();
    let width = 0.5;
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
    f.width_mm=180.; f.height_mm=84.;
    let top=ymax*3.;
    let mut axes=Axes::new([-limit,limit],[baseline,top],"Apparent path residual (m)","Measurements per 0.5 m bin");
    if catalogue {
        axes.set_yscale(AxisScale::Log10)?;
    } else {axes.yticks=(0..=ymax as usize).map(|x|x as f64).collect();}
    let mut p=Panel::new(Rect::new(22.,6.,150.,58.),axes);
    let mut entries=Vec::new();
    let out=hdf5::File::create(folder.join("delay_residual_histogram.h5"))?;
    out.new_dataset_builder().with_data(&residual).create("model_minus_observed_m")?;
    out.new_dataset_builder().with_data(&station).create("station")?;
    out.new_dataset_builder().with_data(&edges).create("bin_edges_m")?;
    let center=median(&residual);
    let mad=median(&residual.iter().map(|r|(r-center).abs()).collect::<Vec<_>>());
    out.new_attr::<f64>().create("median_residual_m")?.write_scalar(&center)?;
    out.new_attr::<f64>().create("unscaled_median_absolute_deviation_m")?.write_scalar(&mad)?;
    println!("All stations: median={center:.6} m, MAD={mad:.6} m");
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
    let mut legend=Legend::new(entries);legend.position=LegendPosition::UpperRight;legend.columns=1;
    // The capped marker spans median +/- the unscaled MAD in data coordinates.
    let marker_y=ymax*1.65;
    let cap=if catalogue {marker_y*0.14} else {ymax*0.08};
    p.push(Layer::Line(Line::new(vec![[center-mad,marker_y],[center+mad,marker_y]],"#171717")));
    for x in [center-mad,center+mad] {
        p.push(Layer::Line(Line::new(vec![[x,marker_y-cap],[x,marker_y+cap]],"#171717")));
    }
    let marker_page_y=if catalogue {64.-58.*(marker_y/baseline).log10()/(top/baseline).log10()}
        else {64.-58.*marker_y/top};
    f.labels.push(Label::new(format!("$2\\,\\mathrm{{MAD}}={:.2}$ m",2.*mad),73.,marker_page_y));
    p.legend=Some(legend);f.panels.push(p);
    f.labels.push(Label::new(format!("All stations: MAD $={mad:.2}$ m"),137.,32.));
    f.latex.cache=folder.parent().unwrap().join(".rustplotlib-cache");
    for ext in ["png","pdf"] {f.save(folder.join(format!("delay_residual_histogram.{ext}")))?;}
    Ok(())
}
