#[path = "../native/common.rs"]
mod common;
use common::*;
use anyhow::Result;
use rustplotlib::*;
use std::path::Path;

fn edges(x: &[f64]) -> Vec<f64> {
    let mut e=vec![x[0]-(x[1]-x[0])/2.];
    e.extend(x.windows(2).map(|p|(p[0]+p[1])/2.));
    e.push(x[x.len()-1]+(x[x.len()-1]-x[x.len()-2])/2.);
    e
}

fn main() -> Result<()> {
    let args:Vec<_>=std::env::args().collect();
    anyhow::ensure!(args.len()==3,"usage: tarnecki-sparse-figure input.h5 output-base");
    let h=hdf5::File::open(&args[1])?;
    let x=array(&h,"velocity_km_s")?;
    let y=array(&h,"log10_n0_rM2_m_inv")?;
    anyhow::ensure!(x.len()==4 && y.len()==4,"Expected exactly a 4x4 grid");
    let ref_x=array(&h,"reference_velocity_km_s")?;
    let ref_y=array(&h,"reference_log10_n0_rM2_m_inv")?;
    let mut f=Figure::new(IntendedUse::ArticleWide);
    f.width_mm=185.;f.height_mm=217.;f.font_pt=9.5;
    let rcs=f.scale(Scale::new([-120.,20.],Normalization::Linear,ColorMap::turbo())?);
    let delta=f.scale(Scale::new([-80.,80.],Normalization::Linear,ColorMap::diverging())?);
    for (col,name) in ["maarsy","eiscat"].iter().enumerate() {
        let g=h.group(name)?;
        for (row,key) in ["reference_full_dbsm","native_fdtd_dbsm","difference_db"].iter().enumerate() {
            let (xx,yy)=if row==0 {(&ref_x,&ref_y)} else {(&x,&y)};
            let values=array(&g,key)?;
            let scale=if row<2 {rcs} else {delta};
            let rect=Rect::new(22.+col as f64*75.,12.+row as f64*67.,60.,49.);
            let mut axes=Axes::new([20.,70.],[y[0],y[3]],"$U$ (km s$^{-1}$)",
                if col==0 {"$\\log_{10}[n_0r_M^2/(\\mathrm{m^{-1}})]$"} else {""});
            axes.xticks=vec![20.,30.,40.,50.,60.,70.];
            axes.yticks=vec![12.4,12.8,13.2,13.6,14.,14.4];
            let mut p=Panel::new(rect,axes);
            p.pcolormesh(edges(xx),edges(yy),values.clone(),scale)?;
            if row==0 {
                let levels:Vec<_>=(-120..=20).step_by(10).map(|v|v as f64).collect();
                let contours=Contour::new(xx.clone(),yy.clone(),values,levels.clone(),scale,false)?;
                for level in levels {
                    let mut pts=Vec::new();
                    for s in contours.segments(level) {pts.extend([s[0],s[1],[f64::NAN;2]]);}
                    let mut l=Line::new(pts,"#ffffff");l.width_pt=0.5;p.push(Layer::Line(l));
                }
            } else {
                let mut px=Vec::new();let mut py=Vec::new();
                for j in 0..4 {for i in 0..4 {if values[4*j+i].is_finite() {px.push(x[i]);py.push(y[j]);}}}
                dots(&mut p,&px,&py,"#111111",2.2,1.,true)?;
            }
            panel_label(&mut f,&p,&format!("{}",(b'a'+(2*row+col) as u8) as char));
            f.panels.push(p);
        }
    }
    f.colorbars.push(Colorbar::new(Rect::new(162.,12.,3.,116.),rcs,"RCS (dBsm)",vec![-120.,-100.,-80.,-60.,-40.,-20.,0.,20.]));
    f.colorbars.push(Colorbar::new(Rect::new(162.,146.,3.,49.),delta,"Difference (dB)",vec![-80.,-40.,0.,40.,80.]));
    let output=Path::new(&args[2]);
    std::fs::create_dir_all(output.parent().unwrap())?;
    f.latex.cache=output.parent().unwrap().join(".native-latex-cache");
    for extension in ["pdf","png","svg"] {f.save(output.with_extension(extension))?;}
    Ok(())
}
