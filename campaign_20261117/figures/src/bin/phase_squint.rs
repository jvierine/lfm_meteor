use anyhow::Result;
use rustplotlib::*;
fn main()->Result<()>{
 let args:Vec<_>=std::env::args().collect();let h=hdf5::File::open(&args[1])?;let z=h.dataset("altitude_km")?.read_raw::<f64>()?;let g=h.dataset("remote_gain_dbi")?.read_raw::<f64>()?;
 let mut f=ArticleLayout::new(177.,10.)?.figure(1.,92.)?;
 for (s,x,c) in [("Header",24.,"#777777"),("Remote correction",80.,"#0072b2"),("All arrays corrected",143.,"#d55e00")]{let mut l=Label::new(s,x,5.);l.color=c.into();f.labels.push(l);}
 for i in 0..2{let x=18.+i as f64*83.;f.labels.push(Label::new(if i==0{"Danzhou"}else{"Wenchang"},x+33.,16.));let mut ax=Axes::new([100.,115.],[15.,43.],"WGS84 altitude (km)",if i==0{"Receive gain (dBi)"}else{""});ax.xticks=vec![100.,105.,110.,115.];ax.yticks=vec![20.,30.,40.];let mut p=Panel::new(Rect::new(x,23.,68.,52.),ax);
 for z0 in [105.,108.]{let mut l=Line::new(vec![[z0,15.],[z0,43.]],"#bcbcbc");l.width_pt=0.5;p.push(Layer::Line(l));}
 for (j,c) in ["#777777","#0072b2","#d55e00"].iter().enumerate(){let base=(j*2+i)*z.len();let mut l=Line::new(z.iter().enumerate().map(|(k,z)|[*z,g[base+k]]).collect(),*c);l.width_pt=1.;p.push(Layer::Line(l));}f.panels.push(p);}
 f.labels.push(Label::new("Fixed-phase model prediction; reference band 105--108 km",90.,89.));
 let out=&args[2];f.save_for_article(format!("{out}.pdf"),&format!("{out}.pdf"))?;f.save(format!("{out}.png"))?;Ok(())
}
