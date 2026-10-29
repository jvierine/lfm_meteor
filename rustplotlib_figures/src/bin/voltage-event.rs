use anyhow::Result;
use rustplotlib::*;
use std::path::Path;

fn limits(x:&[f64], minimum:f64)->[f64;2] {
    let lo=x.iter().copied().fold(f64::INFINITY,f64::min);
    let hi=x.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    let span=(hi-lo).max(minimum);
    [(lo+hi-span)/2.-span*0.08,(lo+hi+span)/2.+span*0.08]
}
fn dots(p:&mut Panel, xy:&[[f64;2]], color:&str)->Result<()> {
    let rgb=[0,1,2].map(|i|u8::from_str_radix(&color[1+2*i..3+2*i],16).unwrap() as f64/255.);
    let c=ColorMap::new(vec![rgb,rgb])?;
    let s=Scale::new([0.,1.],Normalization::Linear,c)?;
    p.scatter_colored(xy,&vec![0.5;xy.len()],&s,3.,0.9,color)
}
fn render(folder:&Path)->Result<()> {
    let h=hdf5::File::open(folder.join("plot.h5"))?;
    let get=|k:&str|->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)};
    let t=get("time_s")?;let station=h.dataset("station")?.read_raw::<i64>()?;
    let observed=get("seed_apparent_path_m")?;let model=get("modeled_path_m")?;
    let beat=get("residual_beat_hz")?;let sigma=get("conditional_beat_std_hz")?;
    let speed=get("speed_km_s")?;let ts=get("speed_time_s")?;
    let mut offset=vec![0.;t.len()];
    for s in 0..3 {let first=(0..t.len()).filter(|&i|station[i]==s).min_by(|&i,&j|t[i].total_cmp(&t[j])).unwrap();
        for i in 0..t.len(){if station[i]==s{offset[i]=(observed[i]-observed[first])/1000.;}}}
    let residual:Vec<_>=model.iter().zip(&observed).map(|(a,b)|a-b).collect();
    let endpoints:Vec<_>=beat.iter().zip(&sigma).flat_map(|(b,s)|[b-1.96*s,b+1.96*s]).collect();
    let ranges=[limits(&offset,0.1),limits(&speed,0.1),limits(&residual,1.),limits(&endpoints,100.)];
    let labels=["Apparent path offset (km)","Speed (km s$^{-1}$)","Apparent path residual (m)","Residual beat (Hz)"];
    let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=220.;f.height_mm=164.;
    let colors=["#0072B2","#D55E00","#009E73"];
    let names=["Sanya","Danzhou","Wenchang"];
    for panel in 0..4 {
        let rect=Rect::new(22.+(panel%2) as f64*110.,25.+(panel/2) as f64*72.,80.,48.);
        let mut p=Panel::new(rect,Axes::new(limits(&t,0.01),ranges[panel],"Time from reference epoch (s)",labels[panel]));
        if panel>=2 {p.push(Layer::Line(Line::new(vec![[t.iter().copied().fold(f64::INFINITY,f64::min),0.],[t.iter().copied().fold(f64::NEG_INFINITY,f64::max),0.]],"#888888")));}
        if panel==1 {
            p.push(Layer::Line(Line::new(ts.iter().zip(&speed).map(|(&x,&y)|[x,y]).collect(),"#222222")));
        } else {
            for s in 0..3 {
                let mut idx:Vec<_>=(0..t.len()).filter(|&i|station[i]==s as i64).collect();idx.sort_by(|&i,&j|t[i].total_cmp(&t[j]));
                let xy:Vec<_>=idx.iter().map(|&i|[t[i],match panel{0=>offset[i],2=>residual[i],_=>beat[i]}]).collect();
                if panel==0 {
                    let base=observed[idx[0]];
                    p.push(Layer::Line(Line::new(idx.iter().map(|&i|[t[i],(model[i]-base)/1000.]).collect(),colors[s])));
                }
                if panel==3 {
                    let bounds:Vec<_>=idx.iter().map(|&i|[beat[i]-1.96*sigma[i],beat[i]+1.96*sigma[i]]).collect();
                    p.errorbars(&xy,None,Some(&bounds),colors[s],0.6,2.,0.7)?;
                }
                dots(&mut p,&xy,colors[s])?;
            }
        }
        if panel==0 {
            let mut legend=Legend::new((0..3).map(|s|LegendEntry::line(names[s],colors[s])).collect());
            legend.position=LegendPosition::AboveCenter;legend.columns=3;
            p.legend=Some(legend);
        }
        f.panels.push(p);
    }
    let id=folder.file_name().unwrap().to_string_lossy().replace('_',"\\_");
    let pass=h.attr("beat_guard_pass")?.read_scalar::<bool>()?;
    f.labels.push(Label::new(format!("{id}: beat check {}",if pass{"PASS"}else{"FAIL"}),110.,5.));
    f.latex.cache=folder.parent().unwrap().join(".rustplotlib-cache");
    for ext in ["pdf","png"] {f.save(folder.join(format!("event_rustplotlib.{ext}")))?;}
    Ok(())
}
fn main()->Result<()> {for arg in std::env::args().skip(1){render(Path::new(&arg))?;}Ok(())}
