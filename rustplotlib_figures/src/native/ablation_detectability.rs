use super::common::*;
use anyhow::Result;
use rustplotlib::*;

pub fn figure(h: &hdf5::File) -> Result<Figure> {
    let threshold=h.attr("threshold_db")?.read_scalar::<f64>()?;
    let frequency=h.attr("frequency_hz").and_then(|a|a.read_scalar::<f64>()).unwrap_or(430e6);
    let elevation=h.attr("elevation_deg").and_then(|a|a.read_scalar::<f64>()).unwrap_or(75.);
    let site_alt=h.attr("station_altitude_m").and_then(|a|a.read_scalar::<f64>()).unwrap_or(50.);
    let fallback_constant=10_f64.powf(1.4);
    let reference_range=-(6371000_f64+50.)*75_f64.to_radians().sin()
        +((6451000_f64).powi(2)-(6371000_f64+50.).powi(2)*75_f64.to_radians().cos().powi(2)).sqrt();
    let link=h.attr("link_constant").and_then(|a|a.read_scalar::<f64>())
        .unwrap_or(fallback_constant*reference_range.powi(4)/1.41669628e-8);
    let mut f=Figure::new(IntendedUse::ArticleWide);
    f.width_mm=175.; f.height_mm=135.; f.font_pt=10.;
    for col in 0..2 {
        let xlabel=if col==0 {"$\\log_{10}(|dm/dt|/[\\mathrm{kg\\,s^{-1}}])$"}
            else {"$\\log_{10}(\\sigma/[\\mathrm{m^2}])$"};
        let mut axes=Axes::new(if col==1 && frequency<1e8 {[-14.,-3.]} else {[-16.,-6.]},[75.,135.],
            xlabel,if col==0 {"Altitude (km)"} else {""});
        axes.xticks=if col==1 && frequency<1e8 {vec![-14.,-12.,-10.,-8.,-6.,-4.]} else {vec![-16.,-14.,-12.,-10.,-8.,-6.]};
        axes.yticks=vec![80.,90.,100.,110.,120.,130.];
        axes.show_y_axis=col==0;
        let mut p=Panel::new(Rect::new(19.+col as f64*80.,38.,71.,75.),axes);
        p.major_grid("#e4e4e4",0.4);
        let mut entries=Vec::new();
        for (j,speed) in [30,50,72].iter().enumerate() {
            let summary=h.group(&format!("summary/v{speed}"))?;
            let path: hdf5::types::VarLenUnicode=summary.attr("profile_path")?.read_scalar()?;
            let altitude:Vec<_>=array(h,&format!("{path}/altitude_m"))?.iter().map(|v|v/1000.).collect();
            let quantity=array(h,&format!("{path}/{}",if col==0 {"mass_loss_rate_kg_s"} else {"rcs_m2"}))?;
            let x:Vec<_>=quantity.iter().map(|v|if *v>0. {v.log10()}else{f64::NAN}).collect();
            curve(&mut p,&x,&altitude,COLORS[j],1.4);
            entries.push(LegendEntry::line(format!("{speed} km s$^{{-1}}$"),COLORS[j]));
        }
        if col==1 {
            let alt:Vec<_>=(75..=135).map(|v|v as f64).collect();
            let limit:Vec<_>=alt.iter().map(|v|{
                let r=-(6371000_f64+site_alt)*elevation.to_radians().sin()
                    +((6371000.+v*1000.).powi(2)-(6371000_f64+site_alt).powi(2)*elevation.to_radians().cos().powi(2)).sqrt();
                (10_f64.powf(threshold/10.)*r.powi(4)/link).log10()
            }).collect();
            curve(&mut p,&limit,&alt,"#777777",1.0);
            entries.push(LegendEntry::line(format!("{threshold:.2} dB threshold"),"#777777"));
        }
        let mut legend=Legend::new(entries);
        legend.position=LegendPosition::AboveCenter;
        legend.columns=1;
        legend.padding_mm=0.; legend.avoid_data=false;
        p.legend=Some(legend);
        f.panels.push(p);
    }
    Ok(f)
}
