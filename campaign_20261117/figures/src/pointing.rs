use super::common::*;
use anyhow::Result;
use rustplotlib::*;

pub fn figure(h: &hdf5::File) -> Result<Figure> {
    let edges = array(h, "edges")?;
    let counts = array(h, "all_counts")?;
    let q = array(h, "all_all_parameters")?;
    let mut f = Figure::new(IntendedUse::Custom { width_mm: 200., font_pt: 10. });
    f.height_mm = 97.;
    let max = counts.iter().copied().fold(1_f64, f64::max);
    let cid = f.scale(Scale::new([1., max], Normalization::Log, ColorMap::inferno())?);
    let wavelength = 299792458. / 430e6;
    for (i, shifted) in [false, true].into_iter().enumerate() {
        let axes = Axes::new([-2.6,2.6],[-2.6,2.6],"East offset (deg)","North offset (deg)");
        let mut p = Panel::new(Rect::new(18.+i as f64*87.,12.,70.,70.),axes);
        p.push(Layer::Pcolormesh(Field::new(edges.clone(),edges.clone(),counts.clone(),cid)?));
        let mut gain=Vec::new();
        for &y in &edges { for &x in &edges {
            let (dx,dy,angle)=if shifted {(q[0],q[1],q[2].to_radians())} else {(0.,0.,0.)};
            let u=(x-dx)*angle.cos()+(y-dy)*angle.sin();
            let v=-(x-dx)*angle.sin()+(y-dy)*angle.cos();
            let az=14.996337890625_f64.to_radians();
            let el=74.9981689453125_f64.to_radians();
            let axis=[az.sin()*el.cos(),az.cos()*el.cos(),el.sin()];
            let mut east=[1.-axis[0]*axis[0],-axis[0]*axis[1],-axis[0]*axis[2]];
            let norm=east.iter().map(|v|v*v).sum::<f64>().sqrt();
            for v in &mut east {*v/=norm;}
            let north=[axis[1]*east[2]-axis[2]*east[1],axis[2]*east[0]-axis[0]*east[2],axis[0]*east[1]-axis[1]*east[0]];
            let mut dir=[0.;3];
            for j in 0..3 {dir[j]=axis[j]+u.to_radians().tan()*east[j]+v.to_radians().tan()*north[j];}
            let norm=dir.iter().map(|v|v*v).sum::<f64>().sqrt();
            let sinc=|v:f64| if v.abs()<1e-12 {1.} else {(std::f64::consts::PI*v).sin()/(std::f64::consts::PI*v)};
            let power=sinc(40./wavelength*(dir[0]/norm-axis[0])).powi(2)*sinc(40./wavelength*(dir[1]/norm-axis[1])).powi(2);
            gain.push(10.*power.max(1e-30).log10());
        }}
        for level in [-20.,-13.,-3.] {
            let c=Contour::new(edges.clone(),edges.clone(),gain.clone(),vec![level,level+1e-9],cid,false)?;
            let mut pts=Vec::new();
            for s in c.segments(level) {pts.extend([s[0],s[1],[f64::NAN;2]]);}
            let mut halo=Line::new(pts.clone(),"#ffffff");halo.width_pt=1.7;p.push(Layer::Line(halo));
            let mut line=Line::new(pts,"#222222");line.width_pt=0.7;p.push(Layer::Line(line));
        }
        panel_label(&mut f,&p,if shifted {"(b) Fitted pattern"} else {"(a) Recorded pointing"});
        f.panels.push(p);
    }
    f.colorbars.push(Colorbar::new(Rect::new(184.,12.,2.,70.),cid,"Count",vec![1.,10.]));
    Ok(f)
}
