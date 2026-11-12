#[allow(dead_code)]
#[path = "../native/common.rs"]
mod common;
use common::*;
use anyhow::Result;
use rustplotlib::*;
use std::path::Path;

fn poly(p: &mut Panel, g: &hdf5::Group, name: &str, color: &str, alpha: f64) -> Result<()> {
    let a = array(g,name)?;
    let mut l=Line::new(a.chunks_exact(2).map(|v|[v[0],v[1]]).collect(),color);
    l.width_pt=0.55;l.fill=Some(color.into());l.fill_opacity=alpha;
    p.push(Layer::Line(l));Ok(())
}
fn main() -> Result<()> {
    let args:Vec<_>=std::env::args().collect();
    anyhow::ensure!(args.len()==3,"usage: memo37-beams data.h5 output-directory");
    let h=hdf5::File::open(&args[1])?;let out=Path::new(&args[2]);
    std::fs::create_dir_all(out)?;
    let heights=array(&h,"altitude_km")?;
    let mut geometry=ArticleLayout::new(178.,10.)?.figure(1.,145.)?;
    let mut gains=ArticleLayout::new(178.,10.)?.figure(1.,88.)?;
    for (i,site) in ["danzhou","wenchang"].iter().enumerate() {
        let g=h.group(site)?;
        let name=if i==0{"Danzhou"}else{"Wenchang"};
        let target=array(&g.group("lower")?,"target")?;
        for zoom in 0..2 {
            let x=17.+87.*i as f64;
            let xr=if zoom==0{[0.,target[0]+18.]}else{[target[0]-8.,target[0]+8.]};
            let yr=if zoom==0{[0.,130.]}else{[80.,120.]};
            let mut p=Panel::new(Rect::new(x,if zoom==0{14.}else{83.},64.,43.),
                Axes::new(xr,yr,"Horizontal distance (km)","Altitude (km)"));
            p.major_grid("#e5e5e5",0.4);
            poly(&mut p,&g,"tx_polygon","#7b3294",0.13)?;
            for (beam,color) in [("lower",COLORS[0]),("upper",COLORS[1])] {
                let b=g.group(beam)?;
                poly(&mut p,&b,"polygon",color,0.17)?;
                let a=array(&b,"axis")?;let mut l=Line::new(a.chunks_exact(2).map(|v|[v[0],v[1]]).collect(),color);
                l.width_pt=0.7;p.push(Layer::Line(l));
                let t=array(&b,"target")?;dots(&mut p,&[t[0]],&[t[1]],color,3.,1.,false)?;
            }
            let full=g.group("full95")?;
            let a=array(&full,"polygon")?;let mut l=Line::new(a.chunks_exact(2).map(|v|[v[0],v[1]]).collect(),"#333333");
            l.width_pt=0.65;l.dash=Some("2 1".into());p.push(Layer::Line(l));
            geometry.labels.push(Label::new(format!("{name}: {}",if zoom==0{"beam paths"}else{"intersection detail"}),x+32.,if zoom==0{7.}else{76.}));
            geometry.panels.push(p);
        }
        let mut p=Panel::new(Rect::new(17.+87.*i as f64,19.,64.,49.),
            Axes::new([-18.,1.],[80.,120.],"Relative receive gain (dB)","Altitude (km)"));
        p.major_grid("#e5e5e5",0.4);
        for (beam,color) in [("full95","#333333"),("lower",COLORS[0]),("upper",COLORS[1])] {
            let y=array(&g.group(beam)?,"gain_relative_full95_db")?;
            let mut l=Line::new(points(&y,&heights),color);l.width_pt=1.25;
            if beam=="full95"{l.dash=Some("2 1".into());}
            p.push(Layer::Line(l));
        }
        gains.labels.push(Label::new(name,49.+87.*i as f64,12.));
        gains.panels.push(p);
    }
    for (f,y) in [(&mut geometry,139.),(&mut gains,82.)] {
        f.labels.push(Label::new("Blue: 95 km half-array; orange: 105 km half-array; dashed: full array at 95 km",89.,y));
        f.latex.cache=out.join(".latex-cache");
    }
    for (f,name) in [(&mut geometry,"memo37_beam_paths"),(&mut gains,"memo37_altitude_gains")] {
        for ext in ["pdf","png"]{f.save(out.join(format!("{name}.{ext}")))?;}
    }
    Ok(())
}
