use super::common::*;
use anyhow::Result;
use rustplotlib::*;

pub fn mass_floor(h: &hdf5::File) -> Result<Figure> {
    let detected=h.attr("mass_epoch").is_ok();
    let y:Vec<_>=array(h,"mass_kg")?.iter().map(|v|v.log10()).collect();
    let rms=array(h,"rms_db")?;let boundary=array(h,"boundary")?;
    let mut f=Figure::new(IntendedUse::ArticleWide);
    f.width_mm=175.;f.height_mm=100.;f.font_pt=9.5;
    for (j,(key,label)) in [("speed_km_s","First-detection speed (km s$^{-1}$)"),
        ("peak_snr_db","Peak main-lobe SNR (dB)")].iter().enumerate() {
        let x=array(h,key)?;
        let mut p=Panel::new(Rect::new(19.+j as f64*85.,15.,64.,66.),
            Axes::new(limits(&x,0.08),limits(&y,0.1),*label,if detected {"$\\log_{10}(m_{\\mathrm{det,RCS}}/\\mathrm{kg})$"} else {"$\\log_{10}(m_{0,\\mathrm{RCS}}/\\mathrm{kg})$"}));
        p.major_grid("#e4e4e4",0.4);
        for good in [false,true] {
            let keep:Vec<_>=(0..x.len()).filter(|i|(rms[*i]<=3. && boundary[*i]<0.5)==good).collect();
            let xx:Vec<_>=keep.iter().map(|i|x[*i]).collect();
            let yy:Vec<_>=keep.iter().map(|i|y[*i]).collect();
            dots(&mut p,&xx,&yy,if good {COLORS[0]} else {"#aaaaaa"},2.8,0.85,false)?;
        }
        f.panels.push(p);
    }
    f.labels.push(Label::new(format!("$N={}$; blue: RCS RMS $\\leq3$ dB, interior solution",y.len()),88.,6.));
    Ok(f)
}

pub fn mass_comparison(h: &hdf5::File) -> Result<Figure> {
    let detected=h.attr("mass_epoch").is_ok();
    let integrated=h.dataset("delta_mass_lo95_noise_only_kg").is_ok();
    let x:Vec<_>=array(h,"dynamic_mass_kg")?.iter().map(|v|v.log10()).collect();
    let y:Vec<_>=array(h,"rcs_mass_kg")?.iter().map(|v|v.log10()).collect();
    let rms=array(h,"rcs_rms_db")?;
    let boundary=array(h,"boundary")?;
    let mut values=x.clone();values.extend(&y);
    let errorbars = integrated && h.dataset("dynamic_lo95_kg").is_ok();
    let mut bounds = Vec::new();
    if errorbars {
        for key in ["dynamic_lo95_kg", "dynamic_hi95_kg", "delta_mass_lo95_noise_only_kg", "delta_mass_hi95_noise_only_kg"] {
            let v: Vec<_> = array(h,key)?.iter().map(|v|v.log10()).collect();
            values.extend(&v); bounds.push(v);
        }
    }
    let lim=[10_f64,1000_f64].map(|r|
        (4. * std::f64::consts::PI / 3. * 3000. * (r * 1e-6).powi(3)).log10());
    let (xl,yl) = (lim,lim);
    let mut f=Figure::new(IntendedUse::ArticleWide);
    f.width_mm=190.;f.height_mm=191.;f.font_pt=10.;
    let mut p=Panel::new(Rect::new(24.,28.,137.,137.),Axes::new(xl,yl,
        if h.attr("marginal_initial_mass").is_ok() {"$\\log_{10}(m_{\\mathrm{dyn}}/\\mathrm{kg})$"} else if detected {"$\\log_{10}(m_{\\mathrm{det,dynamic}}/\\mathrm{kg})$"} else {"$\\log_{10}(m_{\\mathrm{dynamic}}/\\mathrm{kg})$ (event median)"},
        if integrated {"$\\log_{10}(m_{\\mathrm{RCS}}/\\mathrm{kg})$"} else if detected {"$\\log_{10}(m_{\\mathrm{det,RCS}}/\\mathrm{kg})$"} else {"$\\log_{10}(m_{0,\\mathrm{RCS}}/\\mathrm{kg})$"}));
    p.axes.as_mut().unwrap().secondary_x = Some(SecondaryAxis::new(
        r"Initial radius $r_0$ ($\mu\mathrm{m}$)",
        vec![10., 30., 100., 300., 1000., 3000., 10000.],
        |r| (4. * std::f64::consts::PI / 3. * 3000. * (r * 1e-6).powi(3)).log10(),
    )?);
    p.axes.as_mut().unwrap().secondary_y = Some(SecondaryAxis::new(
        r"RCS-equivalent radius ($\mu\mathrm{m}$)",
        vec![10., 30., 100., 300., 1000., 3000., 10000.],
        |r| (4. * std::f64::consts::PI / 3. * 3000. * (r * 1e-6).powi(3)).log10(),
    )?);
    p.major_grid("#e4e4e4",0.4);
    let equality=[xl[0].max(yl[0]),xl[1].min(yl[1])];
    curve(&mut p,&equality,&equality,"#777777",0.8);
    if errorbars {
        let uncertainty = |p: &mut Panel, x: &[f64], y: &[f64]| {
            let mut line = Line::new(points(x, y), "#000000");
            line.width_pt = 0.7;
            line.opacity = 0.5;
            p.push(Layer::Line(line));
        };
        for i in 0..x.len() {
            if bounds[0][i].is_finite() && bounds[1][i].is_finite() {
                uncertainty(&mut p,&[bounds[0][i],bounds[1][i]],&[y[i],y[i]]);
                let cap=(yl[1]-yl[0])*0.005;
                for endpoint in [bounds[0][i],bounds[1][i]] {
                    uncertainty(&mut p,&[endpoint,endpoint],&[y[i]-cap,y[i]+cap]);
                }
            }
            if bounds[2][i].is_finite() && bounds[3][i].is_finite() {
                uncertainty(&mut p,&[x[i],x[i]],&[bounds[2][i],bounds[3][i]]);
                let cap=(xl[1]-xl[0])*0.005;
                for endpoint in [bounds[2][i],bounds[3][i]] {
                    uncertainty(&mut p,&[x[i]-cap,x[i]+cap],&[endpoint,endpoint]);
                }
            }
        }
    }
    for good in [false,true] {
        if integrated && !good {continue;}
        let keep:Vec<_>=(0..x.len()).filter(|i|integrated || (rms[*i]<=3. && boundary[*i]<0.5)==good).collect();
        let xx:Vec<_>=keep.iter().map(|i|x[*i]).collect();
        let yy:Vec<_>=keep.iter().map(|i|y[*i]).collect();
        dots(&mut p,&xx,&yy,if errorbars {"#000000"} else if good {COLORS[0]} else {"#aaaaaa"},if errorbars {2.2} else {3.},if errorbars {1.0} else {0.8},false)?;
    }
    let good=(0..x.len()).filter(|i|rms[*i]<=3. && boundary[*i]<0.5).count();
    if !integrated {
        f.labels.push(Label::new(format!("$N={}$; blue: RCS RMS $\\leq3$ dB, interior ($n={good}$)",x.len()),92.,7.));
    }
    f.panels.push(p);Ok(f)
}

pub fn density_comparison(h: &hdf5::File) -> Result<Figure> {
    let mut names=h.member_names()?;
    names.sort();
    let mut f=Figure::new(IntendedUse::ArticleWide);
    f.width_mm=175.; f.height_mm=190.; f.font_pt=8.5;
    for (i,name) in names.iter().take(4).enumerate() {
        let root=h.group(name)?;
        let g=root.group("rho3000")?;
        let alt:Vec<_>=array(&g,"observed_altitude_m")?.iter().map(|v|v/1000.).collect();
        let obs=array(&g,"observed_rcs_dbsm")?;
        let err=array(&g,"ideal_power_noise_std_db")?;
        let mut xs:Vec<_>=obs.iter().zip(&err).flat_map(|(v,e)|[v-e,v+e]).collect();
        for rho in [3000,300] { xs.extend(array(&root.group(&format!("rho{rho}"))?,"best/predicted_rcs_dbsm")?); }
        let mut yl=limits(&alt,0.1); yl[1]+=0.55*(yl[1]-yl[0]);
        let rect=Rect::new(19.+(i%2) as f64*84.,18.+(i/2) as f64*88.,65.,64.);
        let mut p=Panel::new(rect,Axes::new(limits(&xs,0.08),yl,"RCS (dBsm)","Altitude (km)"));
        p.major_grid("#e4e4e4",0.4);
        for (j,rho) in [3000,300].iter().enumerate() {
            let q=root.group(&format!("rho{rho}"))?;
            curve(&mut p,&array(&q,"best/predicted_rcs_dbsm")?,&alt,COLORS[j+1],1.4);
            let m=q.attr("initial_mass_kg")?.read_scalar::<f64>()?;
            let d=2e6*q.attr("initial_radius_m")?.read_scalar::<f64>()?;
            let rms=q.attr("rms_rcs_residual_db")?.read_scalar::<f64>()?;
            let e=m.log10().floor() as i32; let a=m/10_f64.powi(e);
            let mut label=Label::new(format!("${rho}$: $m_0={a:.2}\\times10^{{{e}}}$ kg, $d_0={d:.0}\\,\\mu$m"),rect.x+32.5,rect.y+4.+j as f64*6.);
            label.color=COLORS[j+1].into();label.background=Some("white".into());label.padding_mm=0.3;
            f.labels.push(label);
            let mut label=Label::new(format!("RMS$_{{{rho}}}={rms:.2}$ dB"),rect.x+32.5,rect.y+16.+j as f64*5.);
            label.color=COLORS[j+1].into();f.labels.push(label);
        }
        for j in 0..alt.len() { let mut l=Line::new(vec![[obs[j]-err[j],alt[j]],[obs[j]+err[j],alt[j]]],COLORS[0]);l.width_pt=0.7;p.push(Layer::Line(l)); }
        dots(&mut p,&obs,&alt,COLORS[0],2.5,1.,false)?;
        f.labels.push(Label::new(name[..8].replace('_',"\\_"),rect.x+32.5,rect.y-3.));
        f.panels.push(p);
    }
    Ok(f)
}

pub fn examples(h: &hdf5::File) -> Result<Figure> {
    let mut names:Vec<_>=h.member_names()?.into_iter().filter(|n|n.starts_with("tri_")).collect();
    names.sort_by(|a,b|{
        let rms=|name:&str|h.group(name).unwrap().attr("rms_rcs_residual_db").unwrap().read_scalar::<f64>().unwrap();
        rms(a).total_cmp(&rms(b))
    });
    let mut f=Figure::new(IntendedUse::ArticleWide);
    f.width_mm=175.; f.height_mm=173.; f.font_pt=9.5;
    for (i,name) in names.iter().take(4).enumerate() {
        let g=h.group(name)?;
        let alt:Vec<_>=array(&g,"observed_altitude_m")?.iter().map(|v|v/1000.).collect();
        let observed=array(&g,"observed_rcs_dbsm")?;
        let predicted=array(&g,"best/predicted_rcs_dbsm")?;
        let error=array(&g,"ideal_power_noise_std_db")?;
        let all:Vec<_>=observed.iter().zip(&error).flat_map(|(v,e)|[v-e,v+e])
            .chain(predicted.iter().copied()).collect();
        let mut altitude_limits=limits(&alt,0.1);
        altitude_limits[1]+=0.25*(altitude_limits[1]-altitude_limits[0]);
        let axes=Axes::new(limits(&all,0.15),altitude_limits,"RCS (dBsm)","Altitude (km)");
        let rect=Rect::new(19.+(i%2) as f64*84.,20.+(i/2) as f64*77.,65.,54.);
        let mut p=Panel::new(rect,axes);
        p.major_grid("#e4e4e4",0.4);
        curve(&mut p,&predicted,&alt,COLORS[1],1.4);
        for j in 0..alt.len() {
            let mut l=Line::new(vec![[observed[j]-error[j],alt[j]],[observed[j]+error[j],alt[j]]],COLORS[0]);
            l.width_pt=0.7; p.push(Layer::Line(l));
        }
        dots(&mut p,&observed,&alt,COLORS[0],2.6,1.,false)?;
        let rms=g.attr("rms_rcs_residual_db")?.read_scalar::<f64>()?;
        f.labels.push(Label::new(format!("{}; RMS = {rms:.2} dB",name[..8].replace('_',"\\_")),rect.x+rect.width/2.,rect.y-3.));
        let mass=g.attr("initial_mass_kg")?.read_scalar::<f64>()?;
        let diameter=2e6*g.attr("initial_radius_m")?.read_scalar::<f64>()?;
        let exponent=mass.log10().floor() as i32;
        let mantissa=mass/10_f64.powi(exponent);
        let mut label=Label::new(format!("$m_0={mantissa:.2}\\times10^{{{exponent}}}$ kg; $d_0={diameter:.0}\\,\\mu$m"),
            rect.x+rect.width/2.,rect.y+4.);
        label.background=Some("white".into());label.padding_mm=0.5;
        f.labels.push(label);
        f.panels.push(p);
    }
    Ok(f)
}

pub fn figure(h: &hdf5::File) -> Result<Figure> {
    let name=h.member_names()?.into_iter().find(|n|n.starts_with("tri_")).unwrap();
    let g=h.group(&name)?;
    let altitude:Vec<_>=array(&g,"observed_altitude_m")?.iter().map(|v|v/1000.).collect();
    let observed=array(&g,"observed_rcs_dbsm")?;
    let predicted=array(&g,"best/predicted_rcs_dbsm")?;
    let error=array(&g,"ideal_power_noise_std_db")?;
    let mut f=Figure::new(IntendedUse::ArticleWide);
    f.width_mm=175.;f.height_mm=102.;f.font_pt=10.;
    let mut axes=Axes::new([-63.,-49.],limits(&altitude,0.12),"RCS (dBsm)","Altitude (km)");
    axes.yticks=vec![88.75,88.80,88.85,88.90,88.95];
    axes.xticks=vec![-60.,-55.,-50.];
    let mut panel=Panel::new(Rect::new(19.,17.,65.,66.),axes);
    panel.major_grid("#e4e4e4",0.4);
    curve(&mut panel,&predicted,&altitude,COLORS[1],1.5);
    for i in 0..altitude.len() {
        let mut line=Line::new(vec![[observed[i]-error[i],altitude[i]],
            [observed[i]+error[i],altitude[i]]],COLORS[0]);
        line.width_pt=0.8;panel.push(Layer::Line(line));
    }
    dots(&mut panel,&observed,&altitude,COLORS[0],3.,1.,false)?;
    let mut legend=Legend::new(vec![LegendEntry::line("Measured RCS",COLORS[0]),
        LegendEntry::line("Best model",COLORS[1])]);
    legend.position=LegendPosition::AboveCenter;legend.columns=1;legend.avoid_data=false;
    panel.legend=Some(legend);f.panels.push(panel);
    let masses=array(&g,"evaluated_log10_mass_kg")?;
    let costs=array(&g,"evaluated_chi_square")?;
    let mut rows:Vec<_>=masses.iter().zip(costs).filter(|(x,_)|**x> -6.5 && **x< -5.)
        .map(|(x,y)|[*x,y.log10()]).collect();
    rows.sort_by(|a,b|a[0].total_cmp(&b[0]));
    let mut axes=Axes::new([-6.5,-5.],[3.,6.],"$\\log_{10}(m_0/\\mathrm{kg})$","$\\log_{10}\\chi^2$");
    axes.xticks=vec![-6.5,-6.,-5.5,-5.];axes.yticks=vec![3.,4.,5.,6.];
    let mut panel=Panel::new(Rect::new(105.,17.,65.,66.),axes);
    panel.major_grid("#e4e4e4",0.4);
    panel.push(Layer::Line(Line::new(rows,"#444444")));
    f.panels.push(panel);
    Ok(f)
}
