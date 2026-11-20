//! Pilot joint path/Doppler likelihood with independent pulse discrepancy.
//! Thermal 2x2 covariances are retained, not rescaled. Two intrinsic scatter
//! scales per station are integrated out, not fixed at residual RMS estimates.
//! Gaussian pulse compression is an approximation requiring coverage checks.
use super::*;
use nalgebra::{DMatrix, DVector, Matrix2, Vector2};
use rand::{SeedableRng, rngs::StdRng};
use std::time::Instant;

struct Node { inverse: Vec<Matrix2<f64>>, constant:f64 }
struct Site { indices:Vec<usize>, nodes:Vec<Node> }
pub(super) struct Discrepancy { measured:Vec<Vector2<f64>>, sites:Vec<Site> }

fn logsum(values:impl Iterator<Item=f64>)->f64 {
    let values:Vec<_>=values.collect();
    let top=values.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    if !top.is_finite(){return top;}
    top+values.iter().map(|x|(x-top).exp()).sum::<f64>().ln()
}

fn scale_grid(n:usize)->Vec<(f64,f64)> {
    let step=(100_f64.ln()+14.)/n as f64;
    let mut grid:Vec<_>=(0..n).map(|j| {
        let log_s=-14.+(j as f64+0.5)*step;let s=log_s.exp();
        // Half-Student-t(3) density times the log-coordinate Jacobian.
        (s,log_s-2.*(1.+s*s/3.).ln()+step.ln())
    }).collect();
    let norm=logsum(grid.iter().map(|x|x.1));
    for x in &mut grid {x.1-=norm;}
    grid
}

impl Discrepancy {
    fn build(v:&Voltage,table:&[f64],station:&[u64],n:usize)->Option<Self> {
        if n<8 || table.len()!=8*station.len(){return None;}
        let rate=v.bandwidth/v.duration/C;
        let shear=Matrix2::new(1.,0.,-rate,1.);
        let mut measured=Vec::new();let mut covariance=Vec::new();
        for row in table.chunks_exact(8) {
            if row[6]!=1. {return None;}
            let c=shear*Matrix2::new(row[3],row[4],row[4],row[5])*shear.transpose();
            if c.cholesky().is_none() || c.iter().any(|x|!x.is_finite()){return None;}
            measured.push(shear*Vector2::new(row[0],row[1]));covariance.push(c);
        }
        let grid=scale_grid(n);let mut sites=Vec::new();
        for site in 0..3 {
            let indices:Vec<_>=station.iter().enumerate().filter_map(|(i,&s)|if s==site {Some(i)}else{None}).collect();
            if indices.is_empty(){continue;}
            let mut nodes=Vec::new();
            for &(a,wa) in &grid {for &(b,wb) in &grid {
                let extra=Matrix2::from_diagonal(&Vector2::new((a*C/v.bandwidth).powi(2),(b/v.duration).powi(2)));
                let mut constant=wa+wb;let mut inverse=Vec::new();
                for &i in &indices {
                    let c=covariance[i]+extra;
                    constant-=0.5*c.determinant().ln();inverse.push(c.try_inverse()?);
                }
                nodes.push(Node{inverse,constant});
            }}
            sites.push(Site{indices,nodes});
        }
        if station.iter().any(|&s|s>2){return None;}
        Some(Self{measured,sites})
    }
    fn logp(&self,v:&Voltage,q:&[f64])->f64 {
        if q.len()!=7 || !(-7. ..=-2.).contains(&q[6]){return f64::NEG_INFINITY;}
        let Some(obs)=v.observables(q) else{return f64::NEG_INFINITY;};
        if obs.len()!=self.measured.len(){return f64::NEG_INFINITY;}
        let residual:Vec<_>=obs.iter().zip(&self.measured).map(|(o,m)|Vector2::new(o[0],o[1])-m).collect();
        self.sites.iter().map(|site|logsum(site.nodes.iter().map(|node| {
            node.constant-0.5*site.indices.iter().zip(&node.inverse).map(|(&i,p)|residual[i].dot(&(p*residual[i]))).sum::<f64>()
        }))).sum()
    }
}

#[no_mangle]
pub unsafe extern "C" fn voltage_discrepancy_new(v:*const Voltage,table:*const f64,
    stations:*const u64,count:usize,nodes:usize)->*mut Discrepancy {
    Discrepancy::build(&*v,std::slice::from_raw_parts(table,8*count),
        std::slice::from_raw_parts(stations,count),nodes).map_or(std::ptr::null_mut(),|x|Box::into_raw(Box::new(x)))
}
#[no_mangle]
pub unsafe extern "C" fn voltage_discrepancy_free(p:*mut Discrepancy) {
    if !p.is_null(){drop(Box::from_raw(p));}
}
#[no_mangle]
pub unsafe extern "C" fn voltage_discrepancy_logp(p:*const Discrepancy,v:*const Voltage,q:*const f64)->f64 {
    (&*p).logp(&*v,std::slice::from_raw_parts(q,7))
}
#[no_mangle]
pub unsafe extern "C" fn voltage_discrepancy_optimize(p:*const Discrepancy,v:*const Voltage,
    initial:*const f64,covariance:*const f64,seconds:f64,out:*mut f64)->i32 {
    let x=DVector::from_column_slice(std::slice::from_raw_parts(initial,7));
    let c=DMatrix::from_row_slice(7,7,std::slice::from_raw_parts(covariance,49));
    let Some(factor)=c.cholesky() else{return 0;};
    let q=importance::newton_mode(|q|(&*p).logp(&*v,q.as_slice()),x,&factor.l(),seconds);
    std::slice::from_raw_parts_mut(out,7).copy_from_slice(q.as_slice());1
}
#[no_mangle]
pub unsafe extern "C" fn voltage_discrepancy_sample(p:*const Discrepancy,v:*const Voltage,
    center:*const f64,covariance:*const f64,draws:usize,seed:u64,path:*const c_char)->i32 {
    let result=(||->anyhow::Result<()> {
        let began=Instant::now();let center=DVector::from_column_slice(std::slice::from_raw_parts(center,7));
        let c=DMatrix::from_row_slice(7,7,std::slice::from_raw_parts(covariance,49));
        let transport=compressed::RadiusTransport::new(&center,&c);
        let proposal=importance::Proposal::new(center,c).ok_or_else(||anyhow::anyhow!("Invalid proposal"))?;
        let mut rng=StdRng::seed_from_u64(seed);let mut samples=Vec::new();let mut weights=Vec::new();
        for _ in 0..draws {
            let z=proposal.draw(&mut rng);let q=transport.physical(&z);
            weights.push((&*p).logp(&*v,q.as_slice())-proposal.logp(&z));samples.extend_from_slice(q.as_slice());
        }
        let h=hdf5::File::create(CStr::from_ptr(path).to_str()?)?;
        h.new_dataset::<f64>().shape((draws,7)).create("samples")?.write_raw(&samples)?;
        h.new_dataset_builder().with_data(&weights).create("log_weights")?;
        h.new_dataset::<f64>().create("elapsed_seconds")?.write_scalar(&began.elapsed().as_secs_f64())?;
        h.flush()?;Ok(())
    })();match result {Ok(())=>1,Err(e)=>{eprintln!("{e:#}");0}}
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn quadrature_prior_normalizes_and_logsum_is_stable() {
        let g=scale_grid(48);assert!(logsum(g.iter().map(|x|x.1)).abs()<1e-12);
        assert!((logsum([1000.,1000.].into_iter())-1000.-2_f64.ln()).abs()<1e-12);
    }
    #[test] fn additive_scatter_preserves_thermal_cross_covariance() {
        let c=Matrix2::new(4.,3.,3.,9.);let extra=Matrix2::from_diagonal(&Vector2::new(16.,25.));
        let total=c+extra;assert_eq!(total[(0,1)],3.);
        assert!(total.determinant()>c.determinant());assert!(total.cholesky().is_some());
    }
}
