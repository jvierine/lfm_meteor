use crate::{config::Config, dataset, metadata, textdata, Pulse};
use anyhow::{ensure, Result};
use hdf5::File;

pub struct NoiseRow {
    utc: i64,
    raw: f64,
    mean: f64,
    mf: f64,
    n: u64,
    gate: f64,
    az: f64,
    el: f64,
    ipp: f64,
}
impl From<&Pulse> for NoiseRow {
    fn from(p: &Pulse) -> Self {
        Self {
            utc: p.utc,
            raw: p.raw_noise,
            mean: p.raw_mean,
            mf: p.noise / std::f64::consts::LN_2,
            n: p.noise_samples,
            gate: p.gate_front,
            az: p.rxaz,
            el: p.rxel,
            ipp: p.ipp,
        }
    }
}
pub fn write(c: &Config, rows: &mut [NoiseRow]) -> Result<()> {
    rows.sort_by_key(|r| r.utc);
    ensure!(
        rows.windows(2).all(|r| r[0].utc != r[1].utc),
        "overlapping timestamps in noise series"
    );
    let tmp = c.output_dir.join("noise.h5.partial");
    let f = File::create(&tmp)?;
    metadata(&f, c)?;
    dataset(
        &f,
        "partial_run",
        &[u8::from(c.max_files > 0 || c.max_frames_per_file > 0)],
    )?;
    dataset(&f, "matched_filter_computed", &[u8::from(!c.noise_only)])?;
    textdata(&f,"noise_method_utf8","Per-pulse raw median(|Q+iI|^2)/ln(2), averaged arithmetically within blocks. Raw mean power is also supplied, without signal rejection. Power units: ADC counts squared per complex sample; no temperature calibration or sky fit applied. Matched-filter median power/ln(2) is in unnormalized matched-filter ADC counts squared. Block boundaries split at time gaps, receiver pointing changes and file-independent count limit. Samples between pulses are not filled.")?;
    let g = f.create_group("pulses")?;
    macro_rules! col {
        ($name:literal,$field:ident) => {
            dataset(
                &g,
                $name,
                &rows.iter().map(|r| r.$field).collect::<Vec<_>>(),
            )?
        };
    }
    col!("utc_ns", utc);
    col!("raw_noise_power_adc2", raw);
    col!("raw_mean_power_adc2", mean);
    col!("matched_filter_noise_power_adc2", mf);
    col!("sample_count", n);
    col!("gate_front_us", gate);
    col!("rx_azimuth_deg", az);
    col!("rx_elevation_deg", el);
    col!("ipp_us", ipp);
    let mut blocks = Vec::new();
    let mut start = 0;
    for i in 1..=rows.len() {
        let boundary = i == rows.len()
            || i - start >= c.noise_average_pulses
            || (i < rows.len()
                && ((rows[i].utc - rows[i - 1].utc) as f64 > rows[i - 1].ipp * 1500.
                    || rows[i].ipp != rows[i - 1].ipp
                    || rows[i].gate != rows[i - 1].gate
                    || (rows[i].az - rows[i - 1].az).abs() > 0.01
                    || (rows[i].el - rows[i - 1].el).abs() > 0.01));
        if boundary {
            blocks.push(&rows[start..i]);
            start = i;
        }
    }
    let g = f.create_group("averages")?;
    dataset(
        &g,
        "start_utc_ns",
        &blocks.iter().map(|b| b[0].utc).collect::<Vec<_>>(),
    )?;
    dataset(
        &g,
        "stop_utc_ns",
        &blocks
            .iter()
            .map(|b| b.last().unwrap().utc)
            .collect::<Vec<_>>(),
    )?;
    dataset(
        &g,
        "utc_ns",
        &blocks
            .iter()
            .map(|b| b[0].utc + (b.last().unwrap().utc - b[0].utc) / 2)
            .collect::<Vec<_>>(),
    )?;
    dataset(
        &g,
        "pulse_count",
        &blocks.iter().map(|b| b.len() as u64).collect::<Vec<_>>(),
    )?;
    for (name, field) in [
        ("raw_noise_power_adc2", 0),
        ("raw_mean_power_adc2", 1),
        ("matched_filter_noise_power_adc2", 2),
    ] {
        dataset(
            &g,
            name,
            &blocks
                .iter()
                .map(|b| {
                    b.iter()
                        .map(|r| match field {
                            0 => r.raw,
                            1 => r.mean,
                            _ => r.mf,
                        })
                        .sum::<f64>()
                        / b.len() as f64
                })
                .collect::<Vec<_>>(),
        )?;
    }
    dataset(
        &g,
        "rx_azimuth_deg",
        &blocks.iter().map(|b| b[0].az).collect::<Vec<_>>(),
    )?;
    dataset(
        &g,
        "rx_elevation_deg",
        &blocks.iter().map(|b| b[0].el).collect::<Vec<_>>(),
    )?;
    f.flush()?;
    drop(g);
    drop(f);
    std::fs::rename(tmp, c.output_dir.join("noise.h5"))?;
    Ok(())
}
