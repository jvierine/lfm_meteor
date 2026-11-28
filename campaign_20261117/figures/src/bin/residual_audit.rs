use anyhow::{ensure, Result};
use hdf5::{types::{FixedAscii, VarLenUnicode}, File, Group};
use std::{collections::HashSet, path::Path, str::FromStr};

#[derive(Default)]
struct Audit {
    residual: Vec<f64>,
    station: Vec<i64>,
    used: Vec<bool>,
    passed: Vec<bool>,
    names: Vec<VarLenUnicode>,
    bad_used: Vec<u64>,
    bad_unused: Vec<u64>,
}

impl Audit {
    fn add(&mut self, name: &str, r: &[f64], s: &[i64], used: &[bool]) -> Result<()> {
        ensure!(r.len() == s.len() && s.len() == used.len());
        let bad = |x: f64| !x.is_finite() || x.abs() >= 40.;
        let n_bad = r.iter().zip(used).filter(|&(x, k)| *k && bad(*x)).count();
        let unused = r.iter().zip(used).filter(|&(x, k)| !*k && bad(*x)).count();
        let pass = used.iter().any(|&k| k) && n_bad == 0;
        self.names.push(VarLenUnicode::from_str(name)?);
        self.bad_used.push(n_bad as u64);
        self.bad_unused.push(unused as u64);
        self.residual.extend_from_slice(r);
        self.station.extend_from_slice(s);
        self.used.extend_from_slice(used);
        self.passed.extend(used.iter().map(|&k| k && pass));
        Ok(())
    }
}

fn read<T: hdf5::H5Type>(h: &Group, key: &str) -> Result<Vec<T>> {
    Ok(h.dataset(key)?.read_raw()?)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 9, "residual_audit APRIL_EVENTS NOVEMBER_AUDIT OUT HEIGHT_INPUT APRIL_HEIGHT NOV17_HEIGHT NOV21_HEIGHT NOV17_CATALOGUE");
    let catalog = File::open(&args[8])?;
    let ids = read::<FixedAscii<32>>(&catalog,"event_id")?;
    let accepted: HashSet<_> = ids.iter().map(|v| v.as_str()).collect();
    let mut a = Audit::default();
    let mut dirs = std::fs::read_dir(&args[1])?.collect::<std::io::Result<Vec<_>>>()?;
    dirs.sort_by_key(|e| e.file_name());
    for p in dirs.into_iter().map(|e| e.path()).filter(|p| p.is_dir()) {
        let h = File::open(p.join("plot.h5"))?;
        let keys = |g: &Group| -> Result<Vec<(i64,i64)>> {
            Ok(read(g,"station")?.into_iter().zip(read(g,"time_ns")?).collect())
        };
        let exclusions = p.join("excluded_measurements.h5");
        let rejected: HashSet<_> = if exclusions.exists() {
            let flags = File::open(exclusions)?;
            keys(&flags)?.into_iter().collect()
        } else { HashSet::new() };
        let used: Vec<_> = keys(&h)?.into_iter().map(|key| !rejected.contains(&key)).collect();
        let r: Vec<_> = read::<f64>(&h,"seed_apparent_path_m")?.into_iter().zip(read::<f64>(&h,"modeled_path_m")?).map(|(o,m)| o-m).collect();
        a.add(&format!("20240422:{}",p.file_name().unwrap().to_string_lossy()), &r,&read(&h,"station")?,&used)?;
    }
    let source = File::open(&args[2])?;
    for day in ["20261117","20261121"] {
        let h = source.group(day)?;
        let r = read(&h,"residual_m")?;
        let s = read(&h,"station")?;
        let used: Vec<_> = read::<u8>(&h,"used")?.into_iter().map(|v| v!=0).collect();
        let names = read::<VarLenUnicode>(&h,"event_id")?;
        let mut start = 0;
        while start < names.len() {
            let stop = start + names[start..].partition_point(|v| v == &names[start]);
            ensure!(stop > start);
            if day != "20261117" || accepted.contains(names[start].as_str()) {
                a.add(&format!("{day}:{}",names[start]), &r[start..stop],&s[start..stop],&used[start..stop])?;
            }
            start = stop;
        }
        let missing = read::<VarLenUnicode>(&h,"missing_fit_mask")?;
        println!("{day} without fit masks: {}",missing.iter().filter(|v| day != "20261117" || accepted.contains(v.as_str())).count());
    }
    let h = File::create(&args[3])?;
    let g = h.create_group("audit")?;
    macro_rules! write { ($group:expr,$key:literal,$val:expr) => { $group.new_dataset_builder().with_data($val).create($key)? }; }
    let r: Vec<_> = a.residual.iter().zip(&a.passed).filter_map(|(&x,&k)| k.then_some(x)).collect();
    let s: Vec<_> = a.station.iter().zip(&a.passed).filter_map(|(&x,&k)| k.then_some(x)).collect();
    write!(h,"residual_m",&r);
    write!(h,"station",&s);
    write!(g,"residual_m",&a.residual);
    write!(g,"station",&a.station);
    write!(g,"used",&a.used);
    write!(g,"plotted",&a.passed);
    write!(g,"event_id",&a.names);
    write!(g,"bad_used",&a.bad_used);
    write!(g,"bad_unused",&a.bad_unused);
    for day in ["20240422","20261117","20261121"] {
        let ids: Vec<_> = a.names.iter().enumerate().filter_map(|(i,n)| n.as_str().starts_with(day).then_some(i)).collect();
        println!("{day}: {} events, {} failed the 40 m gate, {} bad used points, {} bad unused points",
            ids.len(), ids.iter().filter(|&&i| a.bad_used[i]>0).count(),
            ids.iter().map(|&i| a.bad_used[i]).sum::<u64>(),ids.iter().map(|&i| a.bad_unused[i]).sum::<u64>());
    }
    println!("Plotted {} points",r.len());
    let h = File::open_rw(Path::new(&args[4]))?;
    if h.link_exists("monostatic_counts") { h.unlink("monostatic_counts")?; }
    let edges = read::<f64>(&h,"height_edges")?;
    let mut heights = Vec::new();
    let april = File::open(&args[5])?;
    heights.extend(read::<f64>(&april,"monostatic_event_median_height_km")?);
    for path in &args[6..8] {
        let source = File::open(path)?;
        heights.extend(read::<f64>(&source,"monostatic_height_km")?);
    }
    let mut bins = vec![0.;edges.len()-1];
    for h in heights {
        if h.is_finite() && h>=edges[0] && h<=edges[edges.len()-1] {
            let i = (edges.partition_point(|&e| e<=h)-1).min(bins.len()-1);
            bins[i]+=1.;
        }
    }
    write!(h,"monostatic_counts",&bins);
    Ok(())
}
