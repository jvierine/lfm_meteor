#[path = "../native/common.rs"]
mod common;
use common::*;
use anyhow::Result;
use rustplotlib::*;
use std::path::Path;

fn bracket(x:&[f64],v:f64)->usize {x.partition_point(|a|*a<=v).saturating_sub(1).min(x.len()-2)}
fn main()->Result<()> {
    let args:Vec<_>=std::env::args().collect();
    anyhow::ensure!(args.len()==3,"input rcs_mass.h5 output base");
    let h=hdf5::File::open(&args[1])?;
    let g=h.group("intervals/0")?;
    let a=array(&g,"altitude_km")?;let v=array(&g,"speed_km_s")?;
    let r=array(&g,"rcs_dbsm")?;let roots=array(&g,"log10_source")?;
    let interp=array(&g,"interpolated_curves_dbsm")?;
    let lut=h.group("lookup")?;
    let ah=array(&lut,"altitude_km")?;let av=array(&lut,"velocity_km_s")?;
    let s=array(&lut,"log10_source")?;let values=array(&lut,"rcs_dbsm")?;
    let valid:Vec<_>=(0..roots.len()).filter(|&i|roots[i].is_finite()).collect();
    anyhow::ensure!(!valid.is_empty(),"No uniquely invertible samples");
    let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=270.;f.height_mm=145.;f.font_pt=11.;
    for (col,k) in [valid[0],valid[valid.len()/2],*valid.last().unwrap()].into_iter().enumerate(){
        let ai=bracket(&ah,a[k]);let vi=bracket(&av,v[k]);
        let corners=[(ai,vi),(ai,vi+1),(ai+1,vi),(ai+1,vi+1)];
        let colors=["#0072b2","#56b4e9","#009e73","#d55e00"];
        let mut ys=Vec::new();
        for &(i,j) in &corners {ys.extend_from_slice(&values[(i*av.len()+j)*s.len()..(i*av.len()+j+1)*s.len()]);}
        let rect=Rect::new(22.+col as f64*86.,16.,64.,80.);
        let mut ax=Axes::new([s[0]-0.08,s[s.len()-1]+0.08],limits(&ys,0.08),"$\\log_{10}[n_0r_M^2/(\\mathrm{m}^{-1})]$","RCS (dBsm)");
        ax.xticks=vec![12.5,13.,13.5,14.];ax.yticks=vec![-120.,-100.,-80.,-60.,-40.,-20.,0.,20.];let mut p=Panel::new(rect,ax);
        for (c,&(i,j)) in corners.iter().enumerate(){
            let y=&values[(i*av.len()+j)*s.len()..(i*av.len()+j+1)*s.len()];
            curve(&mut p,&s,y,colors[c],0.7);dots(&mut p,&s,y,colors[c],4.,1.,true)?;
            let mut l=Label::new(format!("{} km, {} km s$^{{-1}}$",ah[i],av[j]),rect.x+3.,114.+c as f64*5.);
            l.color=colors[c].into();f.labels.push(l);
        }
        curve(&mut p,&s,&interp[k*s.len()..(k+1)*s.len()],"#222222",1.7);
        dots(&mut p,&[roots[k]],&[r[k]],"#b51e30",6.,1.,false)?;
        f.labels.push(Label::new(format!("{:.2} km, {:.2} km s$^{{-1}}$",a[k],v[k]),rect.x+rect.width/2.,8.));
        f.panels.push(p);
    }
    let out=Path::new(&args[2]);std::fs::create_dir_all(out.parent().unwrap())?;
    f.latex.cache=out.parent().unwrap().join(".native-latex-cache");
    for ext in ["png","pdf","svg"] {f.save(out.with_extension(ext))?;}
    Ok(())
}
