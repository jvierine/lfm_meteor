use super::common::*;
use anyhow::Result;
use rustplotlib::*;

pub fn figure(h: &hdf5::File) -> Result<Figure> {
    let mut f=Figure::new(IntendedUse::ArticleWide);
    f.width_mm=175.;f.height_mm=128.;f.font_pt=10.;
    let mut axes=Axes::new([16.,23.],[-120.,20.],"$\\log_{10}(\\dot N/\\mathrm{s^{-1}})$","RCS (dBsm)");
    axes.xticks=(16..=23).map(|v|v as f64).collect();
    axes.yticks=(-120..=20).step_by(20).map(|v|v as f64).collect();
    let mut p=Panel::new(Rect::new(21.,24.,146.,85.),axes);
    p.major_grid("#e4e4e4",0.4);
    for (name,color) in [("do","#111111"),("pic","#cc00bb")] {
        let g=h.group(name)?;
        let rate=array(&g,"ablation_rate_particles_s")?;
        let x:Vec<_>=rate.iter().map(|v|v.log10()).collect();
        let y:Vec<_>=array(&g,"fdtd_rcs_m2")?.iter().map(|v|10.*v.log10()).collect();
        dots(&mut p,&x,&y,color,3.,1.,true)?;
        let b=g.attr("power_exponent")?.read_scalar::<f64>()?;
        let a=g.attr("power_prefactor")?.read_scalar::<f64>()?;
        let xx=[16.,23.];let yy=xx.map(|v|10.*a.log10()+10.*b*v);
        curve(&mut p,&xx,&yy,color,0.8);
        if name=="do" {
            let radius=array(&g,"archived_radius_m")?;
            let prediction=array(&g,"sphere_rcs_m2")?;
            let keep:Vec<_>=(0..x.len()).filter(|i|radius[*i]>0.).collect();
            let xx:Vec<_>=keep.iter().map(|i|x[*i]).collect();
            let yy:Vec<_>=keep.iter().map(|i|10.*prediction[*i].log10()).collect();
            curve(&mut p,&xx,&yy,COLORS[1],1.4);
            dots(&mut p,&xx,&yy,COLORS[1],2.7,1.,false)?;
        }
    }
    let mut entries=vec![LegendEntry::line("Archived DO FDTD","#111111"),
        LegendEntry::line("Archived PIC FDTD","#cc00bb"),
        LegendEntry::line("Our RCS formula, archived DO radii",COLORS[1])];
    if h.link_exists("native_rust") {
        let g=h.group("native_rust")?;
        let x:Vec<_>=array(&g,"ablation_rate_particles_s")?.iter().map(|v|v.log10()).collect();
        let y:Vec<_>=array(&g,"rcs_m2")?.iter().map(|v|10.*v.log10()).collect();
        dots(&mut p,&x,&y,"#007aaf",5.,1.,false)?;
        entries.push(LegendEntry::circle("Independent Maxwell solve (pilot)","#007aaf"));
    }
    let mut legend=Legend::new(entries);
    legend.position=LegendPosition::AboveCenter;legend.columns=1;legend.avoid_data=false;
    p.legend=Some(legend);f.panels.push(p);Ok(f)
}
