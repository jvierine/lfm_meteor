use anyhow::Result;
use rustplotlib::*;
use std::path::PathBuf;

fn main()->Result<()> {
    let input=PathBuf::from(std::env::args().nth(1).expect("comparison HDF5"));
    let output=PathBuf::from(std::env::args().nth(2).expect("output stem"));
    let h=hdf5::File::open(input)?;
    let events=["tri_0009_1713797502419349670","tri_0189_1713825211644350052",
        "tri_0212_1713827261014350891","tri_0063_1713813402619350433",
        "tri_0110_1713819115444351196","tri_0012_1713798481419350028"];
    let mut figure=ArticleLayout::new(184.,10.)?.figure(1.,137.)?;
    figure.latex.cache=output.parent().unwrap().join(".rustplotlib-cache");
    for (i,event) in events.iter().enumerate() {
        let g=h.group(event)?;
        let old=g.dataset("old_logmass")?.read_raw::<f64>()?;
        let new=g.dataset("new_logmass")?.read_raw::<f64>()?;
        let probability=g.dataset("probability")?.read_raw::<f64>()?;
        let low=old[0].min(new[0]);let high=old[old.len()-1].max(new[new.len()-1]);
        let span=(high-low).max(0.1);let limits=[low-0.06*span,high+0.06*span];
        let rect=Rect::new(17.+(i%3) as f64*59.,9.+(i/3) as f64*60.,43.,41.);
        let mut panel=Panel::new(rect,Axes::new(limits,[0.,1.],r"$\log_{10}(m_0/\mathrm{kg})$",
            if i%3==0 {"Cumulative probability"}else{""}));
        for level in [0.025,0.975] {
            let mut line=Line::new(vec![[limits[0],level],[limits[1],level]],"#cccccc");
            line.dash=Some("2 2".into());panel.push(Layer::Line(line));
        }
        panel.push(Layer::Line(Line::new(old.iter().zip(&probability).map(|(&x,&y)|[x,y]).collect(),"#0072B2")));
        panel.push(Layer::Line(Line::new(new.iter().zip(&probability).map(|(&x,&y)|[x,y]).collect(),"#D55E00")));
        figure.labels.push(Label::new(format!("{}) Event {}",(b'a'+i as u8) as char,event.split('_').nth(1).unwrap()),
            rect.x+rect.width/2.,rect.y-4.));
        if i==2 {
            let mut legend=Legend::new(vec![LegendEntry::line("Noise only","#0072B2"),
                LegendEntry::line("Pulse posterior","#D55E00")]);
            legend.position=LegendPosition::LowerRight;legend.padding_mm=0.5;legend.avoid_data=false;
            panel.legend=Some(legend);
        }
        figure.panels.push(panel);
    }
    figure.save_for_article(output.with_extension("pdf"),"figures/pulse_posterior_comparison.pdf")?;
    Ok(())
}
