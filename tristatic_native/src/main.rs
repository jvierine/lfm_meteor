use anyhow::{bail, Result};
use mpi::traits::*;
use tristatic_native::{event::Event,metropolis};

fn run()->Result<()> {
    let universe=mpi::initialize().ok_or_else(||anyhow::anyhow!("MPI initialization failed"))?;
    let world=universe.world();
    let args:Vec<String>=std::env::args().collect();
    if args.len()!=6 {bail!("usage: tristatic-native INPUT.h5 OUTPUT_DIR WARMUP DRAWS SEED");}
    let warmup=args[3].parse()?; let draws=args[4].parse()?; let seed:u64=args[5].parse()?;
    std::fs::create_dir_all(&args[2])?;
    let event=Event::load(&args[1])?;
    let start=std::time::Instant::now();
    // Each MPI rank is an independent chain; never share an HDF5 writer.
    let chain=metropolis(|q|event.logp(q),&event.initial,&event.widths,warmup,draws,seed+world.rank() as u64)?;
    let output=format!("{}/chain_{:04}.h5",args[2],world.rank());
    if std::path::Path::new(&output).exists() {bail!("Refusing to overwrite {output}");}
    let partial=format!("{output}.partial");
    let h=hdf5::File::create(&partial)?;
    let flat:Vec<f64>=chain.samples.iter().flatten().copied().collect();
    h.new_dataset::<f64>().shape((draws,16)).create("samples")?.write_raw(&flat)?;
    h.new_dataset_builder().with_data(&chain.log_probability).create("log_probability")?;
    h.new_dataset::<u64>().create("accepted")?.write_scalar(&(chain.accepted as u64))?;
    h.new_dataset::<u64>().create("warmup")?.write_scalar(&(warmup as u64))?;
    h.new_dataset::<u64>().create("seed")?.write_scalar(&(seed+world.rank() as u64))?;
    h.new_dataset::<u8>().create("provisional")?.write_scalar(&1)?;
    h.new_dataset::<f64>().create("elapsed_s")?.write_scalar(&start.elapsed().as_secs_f64())?;
    h.new_dataset_builder().with_data(args[1].as_bytes()).create("input_path_utf8")?;
    h.new_dataset::<f64>().shape((16,16)).create("proposal_cholesky_column_major")?
        .write_raw(chain.proposal_cholesky.as_slice())?;
    h.flush()?; drop(h); std::fs::rename(partial,&output)?;
    println!("rank={} draws={} accepted={} elapsed_s={:.3} output={output}",
        world.rank(),draws,chain.accepted,start.elapsed().as_secs_f64());
    Ok(())
}
fn main() {if let Err(e)=run() {eprintln!("{e:#}");std::process::exit(1);}}
