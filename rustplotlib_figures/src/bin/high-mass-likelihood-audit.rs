use anyhow::Result;
use rustplotlib::*;
use std::path::PathBuf;

fn values(h:&hdf5::Group,key:&str)->Result<Vec<f64>> {Ok(h.dataset(key)?.read_raw::<f64>()?)}
fn range(a:&[f64])->[f64;2] {
    let lo=a.iter().copied().fold(f64::INFINITY,f64::min);
    let hi=a.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    [lo-.08*(hi-lo),hi+.08*(hi-lo)]
}
fn line(panel:&mut Panel,x:&[f64],y:&[f64],color:&str) {
    panel.push(Layer::Line(Line::new(x.iter().zip(y).map(|(&x,&y)|[x,y]).collect(),color)));
}

fn main()->Result<()> {
    let input=PathBuf::from(std::env::args().nth(1).expect("audit HDF5"));
    let output=PathBuf::from(std::env::args().nth(2).expect("output stem"));
    let h=hdf5::File::open(input)?;
    let mut figure=ArticleLayout::new(184.,10.)?.figure(1.,190.)?;
    let events=["tri_0009_1713797502419349670","tri_0189_1713825211644350052"];
    for (j,event) in events.iter().enumerate() {
        let g=h.group(event)?;let x=18.+j as f64*92.;let width=64.;
        figure.labels.push(Label::new(format!("Event {}",event.split('_').nth(1).unwrap()),x+width/2.,4.));
        let t=values(&g,"speed_time_s")?;let v=values(&g,"speed_km_s")?;
        let lo=values(&g,"speed_lower95_km_s")?;let hi=values(&g,"speed_upper95_km_s")?;
        let tx=[0.,*t.last().unwrap()*1.03];
        let mut ys=lo.clone();ys.extend(&hi);
        let mut panel=Panel::new(Rect::new(x,12.,width,36.),Axes::new(tx,range(&ys),"Time (s)",r"Speed ($\mathrm{km\,s^{-1}}$)"));
        panel.uncertainty_band(&t,&lo,&hi,"#777777",.5)?;line(&mut panel,&t,&v,"#111111");
        figure.panels.push(panel);
        let station=g.dataset("station")?.read_raw::<i64>()?;let times=values(&g,"time_s")?;
        let statistics=values(&g,"independent_waveform_statistics")?;
        let mut index:Vec<_>=(0..times.len()).filter(|&i|station[i]==0).collect();
        index.sort_by(|&i,&j|times[i].total_cmp(&times[j]));
        let pulse_t:Vec<_>=index.iter().map(|&i|times[i]).collect();
        let echo:Vec<_>=index.iter().map(|&i|statistics[9*i]/(statistics[9*i+1]-2.)).collect();
        let noise:Vec<_>=index.iter().map(|&i|statistics[9*i+2]/statistics[9*i+3]).collect();
        let mut axes=Axes::new(tx,[.7,100.],"Time (s)",r"$P_{\mathrm{res}}/P_{\mathrm{noise}}$");
        axes.set_yscale(AxisScale::Log10)?;axes.yticks=vec![1.,10.,100.];
        let mut panel=Panel::new(Rect::new(x,73.,width,36.),axes);
        line(&mut panel,&tx,&[1.,1.],"#cccccc");line(&mut panel,&pulse_t,&echo,"#D55E00");line(&mut panel,&pulse_t,&noise,"#555555");
        if j==0 {
            let mut legend=Legend::new(vec![LegendEntry::line("Echo residual","#D55E00"),LegendEntry::line("Off-echo noise","#555555")]);
            legend.position=LegendPosition::UpperRight;legend.padding_mm=.5;legend.avoid_data=false;
            panel.legend=Some(legend);
        }
        figure.panels.push(panel);
        let best=g.attr("best_mass_kg")?.read_scalar::<f64>()?;
        let mass=values(&g,"initial_mass_kg_quantiles")?;
        let subset:Vec<_>=values(&g,"omit_sixth_mass_kg")?.iter().map(|m|m/best).collect();
        let mut axes=Axes::new(tx,[.1,10.],"Omitted interval centre (s)",r"$m_{\mathrm{sub}}/m_{\mathrm{all}}$");
        axes.set_yscale(AxisScale::Log10)?;axes.yticks=vec![.1,1.,10.];
        let mut panel=Panel::new(Rect::new(x,134.,width,36.),axes);
        panel.uncertainty_band(&tx,&[mass[0]/best;2],&[mass[2]/best;2],"#777777",.5)?;
        line(&mut panel,&tx,&[1.,1.],"#777777");
        let min_t=times.iter().copied().fold(f64::INFINITY,f64::min);
        let max_t=times.iter().copied().fold(f64::NEG_INFINITY,f64::max);
        let points:Vec<_>=(0..6).map(|i|[min_t+(i as f64+.5)*(max_t-min_t)/6.,subset[i]]).collect();
        panel.markers(&mut figure,&points,&MarkerStyle{shape:MarkerShape::Circle,size_pt:4.,color:[.84,.37,0.]})?;
        figure.panels.push(panel);
    }
    figure.latex.cache=output.parent().unwrap().join(".rustplotlib-cache");
    figure.save_for_article(output.with_extension("pdf"),"figures/high_mass_likelihood_audit.pdf")?;
    Ok(())
}
