use super::common::*;
use anyhow::Result;
use rustplotlib::*;

fn attribute(g: &hdf5::Group, key: &str) -> Result<f64> {
    Ok(g.attr(key)?.read_scalar::<f64>()?)
}
fn legend(p: &mut Panel, entries: Vec<LegendEntry>) {
    let mut l=Legend::new(entries);l.position=LegendPosition::LowerLeft;
    l.avoid_data=false;l.allow_reused_marker_shapes=true;l.background_opacity=0.85;l.padding_mm=1.;p.legend=Some(l);
}
pub fn figure(h: &hdf5::File) -> Result<Figure> {
    let compact = h.attr("compact_article_layout").ok().and_then(|a|a.read_scalar::<bool>().ok()).unwrap_or(false);
    let mut f=Figure::new(IntendedUse::Custom{width_mm:220.,font_pt:10.});
    f.height_mm=if compact {135.} else {248.};
    let t=array(h,"time_s")?;let sites=array(h,"station")?;
    let observed=array(h,"seed_apparent_path_m")?;let model=array(h,"modeled_path_m")?;
    let lower=array(h,"path_lower95_m")?;let upper=array(h,"path_upper95_m")?;
    let ml=array(h,"path_measurement_lower95_m")?;let mh=array(h,"path_measurement_upper95_m")?;
    let beat=array(h,"residual_beat_hz")?;let bs=array(h,"conditional_beat_std_hz")?;
    let tx=limits(&t,0.05);
    for j in 0..4 {
        let rect=Rect::new(if j%2==0{23.}else{126.},if compact {10.+(j/2) as f64*40.} else {12.+(j/2) as f64*77.},73.,if compact {25.} else {54.});
        let mut ys=Vec::new();
        let groups:Vec<Vec<usize>>=(0..3).map(|s|(0..t.len()).filter(|&i|sites[i]==s as f64).collect()).collect();
        let offset=|i:usize|observed[groups[sites[i] as usize][0]];
        for i in 0..t.len() {match j {0=>ys.extend([(ml[i]-offset(i))/1000.,(mh[i]-offset(i))/1000.]),2=>ys.extend([model[i]-mh[i],model[i]-ml[i]]),3=>ys.extend([beat[i]-1.96*bs[i],beat[i]+1.96*bs[i]]),_=>()}}
        let mut yl=if j==1{let mut a=array(h,"speed_lower95_km_s")?;a.extend(array(h,"speed_upper95_km_s")?);limits(&a,0.1)}else{limits(&ys,0.12)};
        if j==1 {yl[1]+=0.35*(yl[1]-yl[0]);}
        let mut p=Panel::new(rect,Axes::new(tx,yl,if compact && j>=2 {""} else {"Time (s)"},["Apparent path offset (km)","Speed (km s$^{-1}$)","Path residual (m)","Residual beat (Hz)"][j]));
        if j==1 {
            let st=array(h,"speed_time_s")?;
            p.uncertainty_band(&st,&array(h,"speed_lower95_km_s")?,&array(h,"speed_upper95_km_s")?,"#777777",0.5)?;
            curve(&mut p,&st,&array(h,"speed_km_s")?,"#222222",1.3);
        }else{
            if j>=2{curve(&mut p,&tx,&[0.,0.],"#aaaaaa",0.6);}
            for (s,ids) in groups.iter().enumerate(){
                let x:Vec<_>=ids.iter().map(|&i|t[i]).collect();
                let y:Vec<_>=ids.iter().map(|&i|match j{0=>(observed[i]-offset(i))/1000.,2=>model[i]-observed[i],_=>beat[i]}).collect();
                if j==0{
                    let lo:Vec<_>=ids.iter().map(|&i|(lower[i]-offset(i))/1000.).collect();let hi:Vec<_>=ids.iter().map(|&i|(upper[i]-offset(i))/1000.).collect();
                    p.uncertainty_band(&x,&lo,&hi,COLORS[s],0.5)?;
                    curve(&mut p,&x,&ids.iter().map(|&i|(model[i]-offset(i))/1000.).collect::<Vec<_>>(),COLORS[s],1.);
                }
                let errors:Vec<_>=ids.iter().map(|&i|match j{0=>[(ml[i]-offset(i))/1000.,(mh[i]-offset(i))/1000.],2=>[model[i]-mh[i],model[i]-ml[i]],_=>[beat[i]-1.96*bs[i],beat[i]+1.96*bs[i]]}).collect();
                p.errorbars(&points(&x,&y),None,Some(&errors),COLORS[s],0.6,1.4,0.8)?;
                dots(&mut p,&x,&y,COLORS[s],2.3,1.,false)?;
            }
            if j==0{legend(&mut p,(0..3).map(|s|LegendEntry::circle(SITES[s],COLORS[s])).collect());}
        }
        panel_label(&mut f,&p,&format!("{})",(b'a'+j as u8) as char));f.panels.push(p);
    }
    let b=h.group("beam")?;let bt=array(&b,"time_s")?;let gain:Vec<_>=array(&b,"relative_gain_db")?.iter().map(|x|2.*x).collect();let rcs=array(&b,"rcs_dbsm")?;
    let good:Vec<_>=(0..bt.len()).filter(|&i|rcs[i].is_finite()&&gain[i]>=-20.).collect();
    let ry:Vec<_>=good.iter().map(|&i|rcs[i]).collect();let rt:Vec<_>=good.iter().map(|&i|bt[i]).collect();
    let mut gy=limits(&gain,0.08);gy[0]-=0.25*(gy[1]-gy[0]);let rl=limits(&ry,0.1);
    let to_gain=|r:f64|gy[0]+(r-rl[0])/(rl[1]-rl[0])*(gy[1]-gy[0]);
    let mut a=Axes::new(limits(&bt,0.05),gy,"Time (s)","Two-way gain (dB)");
    let ticks=(rl[0].ceil() as i32..=rl[1].floor() as i32).filter(|v|if compact {v.rem_euclid(5)==0} else {(v-rl[0].ceil() as i32).rem_euclid(2)==0}).map(|v|v as f64).collect();
    a.secondary_y=Some(SecondaryAxis::new("RCS (dBsm)",ticks,to_gain)?);
    let bottom = if compact {90.} else {166.};
    let panel_height = if compact {25.} else {54.};
    let mut p=Panel::new(Rect::new(23.,bottom,65.,panel_height),a);
    curve(&mut p,&bt,&gain,COLORS[0],1.2);curve(&mut p,&tx,&[-20.,-20.],"#999999",0.6);
    let mapped:Vec<_>=ry.iter().copied().map(to_gain).collect();curve(&mut p,&rt,&mapped,COLORS[1],0.8);dots(&mut p,&rt,&mapped,COLORS[1],2.5,1.,false)?;
    if compact {
        let mut key=Panel::new(Rect::new(23.,81.,65.,6.),Axes::new([0.,65.],[0.,6.],"",""));
        key.axes_visible=false;
        curve(&mut key,&[2.,9.],&[3.,3.],COLORS[0],1.2);
        dots(&mut key,&[47.5],&[3.],COLORS[1],2.5,1.,false)?;
        f.labels.push(Label::new("Two-way gain",50.,84.));
        f.labels.push(Label::new("RCS",84.,84.));
        f.panels.push(key);
    } else {
        legend(&mut p,vec![LegendEntry::line("Two-way gain",COLORS[0]),LegendEntry::circle("RCS",COLORS[1])]);
    }
    panel_label(&mut f,&p,"e)");f.panels.push(p);
    let extent=attribute(&b,"extent_m")?;
    let mut p=Panel::new(Rect::new(126.,bottom,54.,panel_height),Axes::new([-extent,extent],[-extent,extent],"East--west offset (m)","North--south offset (m)"));
    let contours=b.group("contours")?;
    for name in contours.member_names()?{let g=contours.group(&name)?;for path in g.member_names()?{let a=array(&g,&path)?;let mut l=Line::new(a.chunks_exact(2).map(|v|[v[0],v[1]]).collect(),"#999999");l.width_pt=0.6;p.push(Layer::Line(l));}}
    let east=array(&b,"east_m")?;let north=array(&b,"north_m")?;let snr=array(&b,"snr_db")?;
    let scale=Scale::new(limits(&snr,0.01),Normalization::Linear,ColorMap::plasma())?;let cid=f.scale(scale.clone());
    curve(&mut p,&east,&north,"#999999",0.6);p.scatter_colored(&points(&east,&north),&snr,&scale,3.,1.,"none")?;
    let ticks=(scale.limits[0].ceil() as i32..=scale.limits[1].floor() as i32).filter(|v|v%10==0).map(|v|v as f64).collect();
    f.colorbars.push(Colorbar::new(Rect::new(184.,bottom,2.,panel_height),cid,"SNR (dB)",ticks));panel_label(&mut f,&p,"f)");f.panels.push(p);
    Ok(f)
}
