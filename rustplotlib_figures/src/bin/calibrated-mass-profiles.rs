use anyhow::Result;
use rustplotlib::*;
use std::path::PathBuf;

fn main()->Result<()> {
    let regions=PathBuf::from(std::env::args().nth(1).expect("confidence root"));
    let pilot=PathBuf::from(std::env::args().nth(2).expect("bootstrap root"));
    let output=PathBuf::from(std::env::args().nth(3).expect("output stem"));
    let comparison=hdf5::File::open(pilot.join("comparison.h5"))?;
    let events=["tri_0009_1713797502419349670","tri_0189_1713825211644350052",
        "tri_0212_1713827261014350891","tri_0063_1713813402619350433",
        "tri_0110_1713819115444351196","tri_0012_1713798481419350028"];
    let mut figure=ArticleLayout::new(184.,10.)?.figure(1.,137.)?;
    figure.latex.cache=output.parent().unwrap().join(".rustplotlib-cache");
    let constant=(4.*std::f64::consts::PI/3.*3000.).log10();
    for (i,event) in events.iter().enumerate() {
        let h=hdf5::File::open(regions.join(event).join("confidence.h5"))?;
        let old=comparison.dataset(&format!("{event}/old_mass_quantiles_kg"))?.read_raw::<f64>()?;
        let bounds=h.dataset("mass_confidence_interval_kg")?.read_raw::<f64>()?;
        let tested=h.dataset("mass_profile_endpoints_kg")?.read_raw::<f64>()?;
        let limited=h.attr("upper_endpoint_boundary_limited")?.read_scalar::<bool>()?;
        let cutoff=h.attr("cutoff")?.read_scalar::<f64>()?;
        let central=hdf5::File::open(pilot.join(event).join("central_fit.h5"))?;
        let q=central.dataset("params")?.read_raw::<f64>()?;
        let mut points:Vec<[f64;2]>=h.dataset("profile_radius_and_likelihood_ratio")?.read_raw::<f64>()?
            .chunks_exact(2).map(|p|[constant+3.*p[0],(p[1]/cutoff).max(0.)]).collect();
        points.push([constant+3.*q[6],0.]);points.sort_by(|a,b|a[0].total_cmp(&b[0]));
        let low=bounds[0].log10();let high=if limited {tested[1].log10()}else{bounds[1].log10()};
        let span=(high-low).max(0.05);let limits=[low-0.12*span,high+0.12*span];
        let rect=Rect::new(17.+(i%3) as f64*59.,9.+(i/3) as f64*60.,43.,41.);
        let mut panel=Panel::new(rect,Axes::new(limits,[0.,2.],r"$\log_{10}(m_0/\mathrm{kg})$",
            if i%3==0 {"Profile loss / cutoff"}else{""}));
        let accepted=[low,if limited {limits[1]}else{high}];
        panel.uncertainty_band(&accepted,&[0.,0.],&[2.,2.],"#D55E00",0.13)?;
        panel.uncertainty_band(&[old[0].log10(),old[2].log10()],&[0.,0.],&[2.,2.],"#0072B2",0.20)?;
        let mut threshold=Line::new(vec![[limits[0],1.],[limits[1],1.]],"#777777");
        threshold.dash=Some("3 2".into());panel.push(Layer::Line(threshold));
        panel.push(Layer::Line(Line::new(points,"#111111")));
        figure.labels.push(Label::new(format!("{}) Event {}",(b'a'+i as u8) as char,event.split('_').nth(1).unwrap()),
            rect.x+rect.width/2.,rect.y-4.));
        if limited {
            figure.labels.push(Label::new("No upper bound",rect.x+rect.width*0.65,rect.y+rect.height*0.85));
        }
        if i==2 {
            let mut legend=Legend::new(vec![LegendEntry::line("Noise-only interval","#0072B2"),
                LegendEntry::line("Calibrated bounds","#D55E00")]);
            legend.position=LegendPosition::UpperRight;legend.padding_mm=0.5;legend.avoid_data=false;
            panel.legend=Some(legend);
        }
        figure.panels.push(panel);
    }
    figure.save_for_article(output.with_extension("pdf"),"figures/calibrated_mass_profiles.pdf")?;
    Ok(())
}
