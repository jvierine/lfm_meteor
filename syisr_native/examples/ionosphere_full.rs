#[path="ionosphere.rs"] #[allow(dead_code)] mod pilot;
use anyhow::{ensure,Result};
use rustplotlib::*;
use std::path::Path;
fn main()->Result<()> {
    let a:Vec<_>=std::env::args().collect();
    ensure!(a.len()==3,"usage: ionosphere_full PRODUCT_DIRECTORY OUTPUT_DIRECTORY");
    let root=Path::new(&a[1]);let dest=Path::new(&a[2]);std::fs::create_dir_all(dest.join("minutes"))?;
    let mut epoch=0;let mut file=0;let mut rows:Vec<Vec<f64>>=Vec::new();let mut times=Vec::new();
    for minute in 0..1000 {
        let prefix=dest.join(format!("minutes/minute_{minute:04}"));
        let (start,last,used)=pilot::process(root,prefix.to_str().unwrap(),epoch,file)?;
        if used==0 {
            if last==853 {break;}
            rows.push(vec![f64::NAN;400]);times.push(start);
            epoch=start+60_000_000_000;file=last;continue;
        }
        let h=hdf5::File::open(prefix.with_extension("h5"))?;
        let p=h.dataset("block_power_sum")?.read_raw::<f64>()?;
        let n=h.dataset("block_sample_count")?.read_raw::<u64>()?;
        let mask=h.dataset("coherent_candidate_mask")?.read_raw::<u8>()?;
        let noise=h.dataset("late_gate_noise_power_by_second")?.read_raw::<f64>()?;
        let pointing=h.dataset("tx_azimuth_elevation_deg")?.read_raw::<f64>()?;
        let ranges=h.dataset("slant_range_km")?.read_raw::<f64>()?;
        let row=(0..400).map(|b| {
            let mut power=0.;let mut bg=0.;
            for s in 0..60 {let j=s*400+b;if mask[j]==0&&n[j]>0&&noise[s].is_finite() {power+=p[j];bg+=n[j] as f64*noise[s];}}
            if bg>0.{power/bg}else{f64::NAN}
        }).collect();
        rows.push(row);times.push(start);
        let temporary=dest.join("rti.tmp.h5");
        let out=hdf5::File::create(&temporary)?;
        let values:Vec<_>=rows.iter().flatten().copied().collect();
        out.new_dataset::<f64>().shape([rows.len(),400]).create("power_over_noise")?.write_raw(&values)?;
        out.new_dataset_builder().with_data(&times).create("window_start_utc_ns")?;
        out.new_dataset_builder().with_data(&ranges).create("slant_range_km")?;
        out.new_dataset_builder().with_data(&pointing).create("tx_azimuth_elevation_deg")?;
        out.new_dataset_builder().with_data(&[0u8]).create("complete")?;
        out.new_dataset_builder().with_data(b"60-second averages, original LFM voltage; meteor intervals and coherent candidate blocks excluded. Total power/noise, latest 200 us of full-pulse matched-filter lags. Incremental product; complete=0 until entire run ends.").create("methods_utf8")?;
        out.close()?;std::fs::rename(&temporary,dest.join("rti.h5"))?;
        let midnight=times[0].div_euclid(86400000000000)*86400000000000;
        let t0=(times[0]-midnight) as f64/3.6e12;
        let t1=(start+60000000000-midnight) as f64/3.6e12;
        let altitude=|r:f64|(6371f64.powi(2)+r*r+2.*6371.*r*pointing[1].to_radians().sin()).sqrt()-6371.;
        let lo=ranges.iter().position(|r|altitude(*r+1.)>=80.).unwrap();
        let hi=ranges.iter().position(|r|altitude(*r-1.)>=600.).unwrap();
        let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=170.;f.height_mm=100.;
        let scale=f.scale(Scale::new([0.95,1.5],Normalization::Linear,ColorMap::storm())?);
        let mut panel=Panel::new(Rect::new(23.,8.,116.,72.),Axes::new([t0,t1],[80.,600.],"UTC hour (2026-11-17)","Altitude (km)"));
        let data:Vec<_>=(lo..hi).flat_map(|b|rows.iter().map(move|row|row[b])).collect();
        panel.push(Layer::Pcolormesh(Field::new((0..=rows.len()).map(|i|t0+i as f64/60.).collect(),(lo..=hi).map(|b|altitude(ranges[b]-1.)).collect(),data,scale)?));
        f.panels.push(panel);f.colorbars.push(Colorbar::new(Rect::new(142.,8.,3.,72.),scale,"Power / noise",vec![1.,1.2,1.4]));
        f.save(dest.join("rti.tmp.png"))?;std::fs::rename(dest.join("rti.tmp.png"),dest.join("rti.png"))?;
        println!("PUBLISHED minute {minute}: {used} pulses, source file {last}, {} minutes in RTI",rows.len());
        epoch=start+60_000_000_000;file=last;
    }
    let done=hdf5::File::open_rw(dest.join("rti.h5"))?;done.dataset("complete")?.write_raw(&[1u8])?;
    println!("COMPLETE {} windows",rows.len());
    Ok(())
}
