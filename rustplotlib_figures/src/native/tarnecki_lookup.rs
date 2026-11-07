use super::common::*;
use anyhow::Result;
use rustplotlib::*;

fn edges(x: &[f64]) -> Vec<f64> {
    let mut e=vec![x[0]-(x[1]-x[0])/2.];
    e.extend(x.windows(2).map(|p|(p[0]+p[1])/2.));
    e.push(x[x.len()-1]+(x[x.len()-1]-x[x.len()-2])/2.);e
}

pub fn figure(h: &hdf5::File) -> Result<Figure> {
    let x=array(h,"velocity_km_s")?;
    let y=array(h,"log10_n0_rM2_m_inv")?;
    let mut f=Figure::new(IntendedUse::ArticleWide);
    f.width_mm=185.;f.height_mm=229.;f.font_pt=9.5;
    let rcs_scale=f.scale(Scale::new([-160.,20.],Normalization::Linear,ColorMap::turbo())?);
    let diff_scale=f.scale(Scale::new([-90.,90.],Normalization::Linear,ColorMap::diverging())?);
    for (col,name) in ["maarsy","eiscat"].iter().enumerate() {
        let g=h.group(name)?;
        for (row,key) in ["approximation_dbsm","tarnecki_fdtd_dbsm","difference_db"].iter().enumerate() {
            let values=array(&g,key)?;
            let scale=if row<2 {rcs_scale} else {diff_scale};
            let rect=Rect::new(21.+col as f64*75.,18.+row as f64*69.,61.,49.);
            let mut axes=Axes::new([20.,70.],[y[0],y[y.len()-1]],"$U$ (km s$^{-1}$)",
                if col==0 {"$\\log_{10}[n_0r_M^2/(\\mathrm{m^{-1}})]$"} else {""});
            axes.xticks=vec![20.,30.,40.,50.,60.,70.];
            axes.yticks=vec![12.4,12.8,13.2,13.6,14.,14.4];
            let mut p=Panel::new(rect,axes);
            p.pcolormesh(edges(&x),edges(&y),values.clone(),scale)?;
            if row<2 {
                let levels:Vec<_>=(-150..=10).step_by(10).map(|v|v as f64).collect();
                let contours=Contour::new(x.clone(),y.clone(),values,levels.clone(),scale,false)?;
                for level in levels {
                    let mut points=Vec::new();
                    for segment in contours.segments(level) {points.extend([segment[0],segment[1],[f64::NAN;2]]);}
                    let mut line=Line::new(points,"#ffffff");line.width_pt=0.55;
                    p.push(Layer::Line(line));
                }
            }
            f.panels.push(p);
            let label=match row {0=>"Rayleigh--geometric",1=>"Tarnecki et al. (2021)",_=>"Approximation minus FDTD"};
            f.labels.push(Label::new(label,rect.x+rect.width/2.,rect.y-3.));
        }
        f.labels.push(Label::new(if col==0 {"MAARSY, 53.5 MHz; 100 km"} else {"EISCAT, 929 MHz; 100 km"},51.5+col as f64*75.,5.));
    }
    f.colorbars.push(Colorbar::new(Rect::new(162.,18.,3.,118.),rcs_scale,"RCS (dBsm)",vec![-160.,-120.,-80.,-40.,0.,20.]));
    f.colorbars.push(Colorbar::new(Rect::new(162.,156.,3.,49.),diff_scale,"Difference (dB)",vec![-90.,-60.,-30.,0.,30.,60.,90.]));
    Ok(f)
}
