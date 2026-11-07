use super::common::*;
use anyhow::Result;
use rustplotlib::*;

pub fn summary(h: &hdf5::File) -> Result<Figure> {
    let g=h.group("summary")?;
    let mass=array(&g,"observed_ablated_mass_kg")?;
    let y:Vec<_>=mass.iter().map(|m|m.log10()).collect();
    let mut f=Figure::new(IntendedUse::ArticleWide);
    f.width_mm=175.;f.height_mm=96.;f.font_pt=10.;
    for (i,(key,label)) in [("speed_km_s","Measured speed (km s$^{-1}$)"),
        ("peak_snr_db","Peak main-lobe SNR (dB)")].iter().enumerate() {
        let x=array(&g,key)?;
        let mut p=Panel::new(Rect::new(19.+i as f64*85.,8.,64.,67.),
            Axes::new(limits(&x,0.08),limits(&y,0.1),*label,"$\\log_{10}(\\Delta m_{\\mathrm{obs}}/\\mathrm{kg})$"));
        p.major_grid("#e4e4e4",0.4);
        let valid:Vec<_>=(0..y.len()).filter(|j|y[*j].is_finite()).collect();
        dots(&mut p,&valid.iter().map(|j|x[*j]).collect::<Vec<_>>(),
            &valid.iter().map(|j|y[*j]).collect::<Vec<_>>(),COLORS[0],2.8,0.85,false)?;
        f.panels.push(p);
    }
    Ok(f)
}

pub fn event(h: &hdf5::File) -> Result<Figure> {
    let name=h.member_names()?.into_iter().find(|n|n.starts_with("tri_")).unwrap();
    let g=h.group(&name)?;
    let alt=array(&g,"altitude_km")?;
    let rate=array(&g,"mass_loss_rate_kg_s")?;
    let lo=array(&g,"rate_lo95_noise_only_kg_s")?;
    let hi=array(&g,"rate_hi95_noise_only_kg_s")?;
    let pairs=g.dataset("accepted_pair")?.read_raw::<bool>()?;
    let mut f=Figure::new(IntendedUse::ArticleWide);
    f.width_mm=175.;f.height_mm=110.;f.font_pt=9.5;
    for i in 0..2 {
        let x=if i==0 {array(&g,"rcs_dbsm")?} else {rate.iter().map(|v|v.log10()).collect()};
        let mut yl=limits(&alt,0.12);yl[1]+=0.2*(yl[1]-yl[0]);
        if yl[1]-yl[0]<0.02 {yl=[alt[0]-0.05,alt[0]+0.05];}
        let mut xl=limits(&x,0.12);
        if xl[1]-xl[0]<0.02 {xl=[x[0]-0.1,x[0]+0.1];}
        let mut p=Panel::new(Rect::new(19.+i as f64*85.,14.,64.,72.),
            Axes::new(xl,yl,if i==0 {"RCS (dBsm)"} else {"$\\log_{10}(|\\dot m|/\\mathrm{kg\\,s^{-1}})$"},"Altitude (km)"));
        p.major_grid("#e4e4e4",0.4);
        for j in 0..pairs.len() {
            if pairs[j] {curve(&mut p,&x[j..j+2],&alt[j..j+2],COLORS[0],0.9);}
        }
        if i==1 {
            for j in 0..rate.len() {
                curve(&mut p,&[lo[j].log10(),hi[j].log10()],&[alt[j],alt[j]],COLORS[0],0.7);
            }
        }
        dots(&mut p,&x,&alt,COLORS[0],2.8,1.,false)?;
        f.panels.push(p);
    }
    let m=g.attr("observed_ablated_mass_kg")?.read_scalar::<f64>()?;
    let text=if m.is_finite() {
        let exponent=m.log10().floor() as i32;
        let mantissa=m/10_f64.powi(exponent);
        let duration=g.attr("covered_duration_s")?.read_scalar::<f64>()?;
        format!("$\\Delta m_{{\\mathrm{{obs}}}}={mantissa:.2}\\times10^{{{exponent}}}$ kg; $\\Delta t={duration:.3}$ s")
    } else {"Rate only; no integration interval".into()};
    let mut label=Label::new(text,136.,20.);
    label.background=Some("white".into());label.padding_mm=0.5;f.labels.push(label);
    Ok(f)
}
