use anyhow::Result;
use rustplotlib::*;
use std::path::{Path,PathBuf};

fn curve(path:&Path,power:f64,lo:f64,hi:f64)->Result<Vec<[f64;2]>> {
    let h=hdf5::File::open(path)?;
    let samples=h.dataset("samples")?.read_raw::<f64>()?;
    let weights=h.dataset("weights")?.read_raw::<f64>()?;
    let mass:Vec<f64>=samples.chunks_exact(7).map(|q|4.*std::f64::consts::PI/3.*3000.*10_f64.powf(3.*q[6]-power)).collect();
    let mean=mass.iter().zip(&weights).map(|(m,w)|m*w).sum::<f64>();
    let variance=mass.iter().zip(&weights).map(|(m,w)|(m-mean).powi(2)*w).sum::<f64>();
    let bandwidth=0.12*variance.sqrt();
    Ok((0..241).map(|j|{
        let x=lo+(hi-lo)*j as f64/240.;
        let density=mass.iter().zip(&weights).filter(|(_,w)|**w>0.).map(|(m,w)|
            w*(-0.5*((x-m)/bandwidth).powi(2)).exp()).sum::<f64>()/(bandwidth*std::f64::consts::TAU.sqrt());
        [x,density]
    }).collect())
}

fn main()->Result<()> {
    let first=PathBuf::from(std::env::args().nth(1).expect("first posterior root"));
    let second=PathBuf::from(std::env::args().nth(2).expect("independent posterior root"));
    let output=PathBuf::from(std::env::args().nth(3).expect("output stem"));
    let events=["tri_0110_1713819115444351196","tri_0112_1713819314694350243","tri_0142_1713821835054349899"];
    let mut figure=ArticleLayout::new(180.,10.)?.figure(1.,74.)?;
    figure.latex.cache=output.parent().unwrap().join(".rustplotlib-cache");
    for (i,event) in events.iter().enumerate() {
        let h=hdf5::File::open(first.join(event).join("posterior.h5"))?;
        let quantile=h.dataset("mass_kg_quantiles")?.read_raw::<f64>()?;
        let power=quantile[1].log10().floor();let scale=10_f64.powf(power);
        let width=quantile[2]-quantile[0];
        let lo=(quantile[0]-0.6*width).max(0.)/scale;let hi=(quantile[2]+0.6*width)/scale;
        let a=curve(&first.join(event).join("posterior.h5"),power,lo,hi)?;
        let b=curve(&second.join(event).join("posterior.h5"),power,lo,hi)?;
        let top=a.iter().chain(&b).map(|p|p[1]).fold(0_f64,f64::max)*1.18;
        let rect=Rect::new(18.+i as f64*56.,8.,43.,48.);
        let label=format!(r"$m_0$ ($10^{{{power:.0}}}$ kg)");
        let mut panel=Panel::new(rect,Axes::new([lo,hi],[0.,top],label,if i==0 {"Posterior density"}else{""}));
        panel.push(Layer::Line(Line::new(a,"#0072B2")));
        let mut repeated=Line::new(b,"#D55E00");repeated.dash=Some("3 2".into());
        panel.push(Layer::Line(repeated));
        if i==0 {
            let mut legend=Legend::new(vec![LegendEntry::line("Run 1","#0072B2"),LegendEntry::line("Run 2","#D55E00")]);
            legend.position=LegendPosition::UpperRight;legend.columns=2;legend.padding_mm=0.5;
            panel.legend=Some(legend);
        }
        figure.labels.push(Label::new(format!("Event {}",event.split('_').nth(1).unwrap()),rect.x+rect.width/2.,3.));
        figure.panels.push(panel);
    }
    for extension in ["pdf","png"] {figure.save(output.with_extension(extension))?;}
    Ok(())
}
