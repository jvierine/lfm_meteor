//! Seven-parameter posterior quadrature: fix log radius, integrate six coordinates.
//! Gaussian proposals only choose integration nodes; every weight uses the voltage likelihood.
use super::*;
use super::compressed::{Cache, simplex_minimize};
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

fn normal_rule(order:usize)->(Vec<f64>,Vec<f64>) {
    let mut jacobi=DMatrix::zeros(order,order);
    for i in 1..order {jacobi[(i-1,i)]=(i as f64).sqrt();jacobi[(i,i-1)]=(i as f64).sqrt();}
    let eig=jacobi.symmetric_eigen();
    let mut indices:Vec<usize>=(0..order).collect();
    indices.sort_by(|&i,&j|eig.eigenvalues[i].total_cmp(&eig.eigenvalues[j]));
    (indices.iter().map(|&i|eig.eigenvalues[i]).collect(),
     indices.iter().map(|&i|eig.eigenvectors[(0,i)].powi(2)).collect())
}

struct Integral {log_value:f64,mean:DVector<f64>,covariance:DMatrix<f64>,evaluations:usize}

fn integrate<F:Fn(&DVector<f64>)->f64>(likelihood:F,center:&DVector<f64>,factor:&DMatrix<f64>,order:usize)->Integral {
    let dimension=center.len();let (nodes,weights)=normal_rule(order);
    let mut maximum=f64::NEG_INFINITY;let mut total=0.;
    let mut first=DVector::zeros(dimension);let mut second=DMatrix::zeros(dimension,dimension);
    let evaluations=order.pow(dimension as u32);
    for index in 0..evaluations {
        let mut remaining=index;let mut z=DVector::zeros(dimension);let mut log_weight=0.;
        for j in 0..dimension {let k=remaining%order;remaining/=order;z[j]=nodes[k];log_weight+=weights[k].ln();}
        let x=center+factor*&z;
        log_weight+=likelihood(&x)+0.5*z.norm_squared();
        if !log_weight.is_finite(){continue;}
        if log_weight>maximum {
            let scale=(maximum-log_weight).exp();total*=scale;first*=scale;second*=scale;maximum=log_weight;
        }
        let weight=(log_weight-maximum).exp();total+=weight;first+=&x*weight;second+=(&x*x.transpose())*weight;
    }
    if total<=0. {return Integral{log_value:f64::NEG_INFINITY,mean:first,covariance:second,evaluations};}
    let mean=first/total;let covariance=second/total-&mean*mean.transpose();
    let log_value=maximum+total.ln()+0.5*dimension as f64*(std::f64::consts::TAU).ln()
        +(0..dimension).map(|j|factor[(j,j)].abs().ln()).sum::<f64>();
    Integral{log_value,mean,covariance,evaluations}
}

#[no_mangle]
pub unsafe extern "C" fn voltage_radius_grid_point(cache:*const Cache,voltage:*const Voltage,
    seed:*const f64,covariance:*const f64,log_radius:f64,order:usize,inflation:f64,
    optimize_seconds:f64,path:*const c_char)->i32 {
    let result=(||->anyhow::Result<()> {
        anyhow::ensure!((1..=15).contains(&order) && inflation>0. && (-7. ..=-2.).contains(&log_radius),"Invalid grid specification");
        let began=Instant::now();let cache=&*cache;let voltage=&*voltage;
        let seed=DVector::from_column_slice(std::slice::from_raw_parts(seed,7));
        let covariance=DMatrix::from_row_slice(7,7,std::slice::from_raw_parts(covariance,49));
        let cross=covariance.view((0,6),(6,1)).into_owned();
        let permutation=[6,0,1,2,3,4,5];
        let reordered=DMatrix::from_fn(7,7,|i,j|covariance[(permutation[i],permutation[j])]);
        let factor=reordered.cholesky().ok_or_else(||anyhow::anyhow!("Invalid conditional metric"))?.l();
        let transform=factor.view((1,1),(6,6)).into_owned();
        let delta=log_radius-seed[6];let beta=&cross/covariance[(6,6)];
        let shift=-(-std::f64::consts::LN_10*delta).exp_m1()/std::f64::consts::LN_10;
        let make=|x:&DVector<f64>| {let mut q=seed.clone();q.rows_mut(0,6).copy_from(x);q[6]=log_radius;q};
        let plain=seed.rows(0,6).into_owned();
        let candidates=[plain.clone(),&plain+beta.column(0)*delta,&plain+beta.column(0)*shift];
        let initial=candidates.into_iter().max_by(|a,b|cache.logp(voltage,make(a).as_slice()).total_cmp(&cache.logp(voltage,make(b).as_slice()))).unwrap();
        let initial_lp=cache.logp(voltage,make(&initial).as_slice());
        let h=hdf5::File::create(CStr::from_ptr(path).to_str()?)?;
        h.new_dataset::<f64>().create("log10_radius_m")?.write_scalar(&log_radius)?;
        h.new_dataset::<u64>().create("order")?.write_scalar(&(order as u64))?;
        h.new_dataset::<f64>().create("proposal_inflation")?.write_scalar(&inflation)?;
        if !initial_lp.is_finite() {
            h.new_dataset::<u8>().create("valid")?.write_scalar(&0)?;
            h.new_dataset::<f64>().create("log_marginal")?.write_scalar(&f64::NEG_INFINITY)?;
            h.flush()?;return Ok(());
        }
        let objective=|z:&DVector<f64>| {let value=cache.logp(voltage,make(&(&initial+&transform*z)).as_slice());
            if value.is_finite(){-value}else{f64::INFINITY}};
        let (z,optimized)=simplex_minimize(objective,6,optimize_seconds);
        let center=&initial+&transform*z;let mode=make(&center);let mode_lp=cache.logp(voltage,mode.as_slice());
        let mut information=DMatrix::zeros(6,6);let zero=DVector::zeros(6);let step=0.12;
        let at=|d:&DVector<f64>|cache.logp(voltage,make(&(&center+&transform*d)).as_slice());
        for j in 0..6 {
            let mut a=zero.clone();a[j]=step;
            information[(j,j)]=-(at(&a)-2.*mode_lp+at(&(-&a)))/(step*step);
            for k in 0..j {
                let mut b=zero.clone();b[k]=step;
                let value=-(at(&(&a+&b))-at(&(&a-&b))-at(&(-&a+&b))+at(&(-&a-&b)))/(4.*step*step);
                information[(j,k)]=value;information[(k,j)]=value;
            }
        }
        let curvature_computed=information.iter().all(|x|x.is_finite());
        let (metric,minimum)=if curvature_computed {
            let eig=information.symmetric_eigen();let minimum=eig.eigenvalues.min();
            let floor=(eig.eigenvalues.max()*1e-8).max(if minimum>0. {1e-6}else{0.05});
            (&eig.eigenvectors*DMatrix::from_diagonal(&eig.eigenvalues.map(|x|1./x.max(floor)))*eig.eigenvectors.transpose(),minimum)
        }else{(DMatrix::identity(6,6),f64::NAN)};
        let proposal=&transform*metric*transform.transpose();
        let factor=proposal.cholesky().ok_or_else(||anyhow::anyhow!("Invalid quadrature proposal"))?.l()*inflation;
        let integral=integrate(|x|cache.logp(voltage,make(x).as_slice()),&center,&factor,order);
        h.new_dataset::<u8>().create("valid")?.write_scalar(&(integral.log_value.is_finite() as u8))?;
        h.new_dataset::<u8>().create("optimizer_converged")?.write_scalar(&(optimized as u8))?;
        h.new_dataset::<f64>().create("profile_log_likelihood")?.write_scalar(&mode_lp)?;
        h.new_dataset::<f64>().create("log_marginal")?.write_scalar(&integral.log_value)?;
        h.new_dataset::<f64>().create("minimum_curvature")?.write_scalar(&minimum)?;
        h.new_dataset::<u8>().create("curvature_computed")?.write_scalar(&(curvature_computed as u8))?;
        h.new_dataset_builder().with_data(mode.as_slice()).create("conditional_mode")?;
        let proposal_flat:Vec<f64>=(0..6).flat_map(|i|(0..6).map(move |j|(i,j))).map(|(i,j)|factor[(i,j)]).collect();
        h.new_dataset::<f64>().shape((6,6)).create("proposal_factor")?.write_raw(&proposal_flat)?;
        h.new_dataset_builder().with_data(integral.mean.as_slice()).create("conditional_mean")?;
        let flat:Vec<f64>=(0..6).flat_map(|i|(0..6).map(move |j|(i,j))).map(|(i,j)|integral.covariance[(i,j)]).collect();
        h.new_dataset::<f64>().shape((6,6)).create("conditional_covariance")?.write_raw(&flat)?;
        h.new_dataset::<u64>().create("evaluations")?.write_scalar(&(integral.evaluations as u64))?;
        h.new_dataset::<f64>().create("elapsed_seconds")?.write_scalar(&began.elapsed().as_secs_f64())?;
        h.flush()?;Ok(())
    })();match result {Ok(())=>1,Err(e)=>{eprintln!("{e:#}");0}}
}

#[no_mangle]
pub unsafe extern "C" fn voltage_radius_grid_integrate(cache:*const Cache,voltage:*const Voltage,
    profile:*const c_char,order:usize,inflation:f64,path:*const c_char)->i32 {
    let result=(||->anyhow::Result<()> {
        anyhow::ensure!((1..=15).contains(&order) && inflation>0.,"Invalid quadrature specification");
        let began=Instant::now();let cache=&*cache;let voltage=&*voltage;
        let input=hdf5::File::open(CStr::from_ptr(profile).to_str()?)?;
        let mode=DVector::from_vec(input.dataset("conditional_mode")?.read_raw::<f64>()?);
        let factor=DMatrix::from_row_slice(6,6,&input.dataset("proposal_factor")?.read_raw::<f64>()?)*inflation;
        let integral=integrate(|x|{let mut q=mode.clone();q.rows_mut(0,6).copy_from(x);cache.logp(voltage,q.as_slice())},
            &mode.rows(0,6).into_owned(),&factor,order);
        let h=hdf5::File::create(CStr::from_ptr(path).to_str()?)?;
        h.new_dataset::<f64>().create("log10_radius_m")?.write_scalar(&mode[6])?;
        h.new_dataset::<u64>().create("order")?.write_scalar(&(order as u64))?;
        h.new_dataset::<f64>().create("proposal_inflation")?.write_scalar(&inflation)?;
        h.new_dataset::<u8>().create("valid")?.write_scalar(&(integral.log_value.is_finite() as u8))?;
        h.new_dataset::<f64>().create("log_marginal")?.write_scalar(&integral.log_value)?;
        h.new_dataset_builder().with_data(integral.mean.as_slice()).create("conditional_mean")?;
        let flat:Vec<f64>=(0..6).flat_map(|i|(0..6).map(move |j|(i,j))).map(|(i,j)|integral.covariance[(i,j)]).collect();
        h.new_dataset::<f64>().shape((6,6)).create("conditional_covariance")?.write_raw(&flat)?;
        h.new_dataset::<u64>().create("evaluations")?.write_scalar(&(integral.evaluations as u64))?;
        h.new_dataset::<f64>().create("elapsed_seconds")?.write_scalar(&began.elapsed().as_secs_f64())?;
        h.flush()?;Ok(())
    })();match result {Ok(())=>1,Err(e)=>{eprintln!("{e:#}");0}}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integrates_six_dimensional_correlated_normal() {
        let mean=DVector::from_vec(vec![0.2,-0.3,0.1,0.7,-0.2,0.4]);
        let mut factor=DMatrix::identity(6,6);
        for j in 0..6 {factor[(j,j)]=0.5+j as f64*0.1;for k in 0..j {factor[(j,k)]=0.1;}}
        let inverse=factor.clone().try_inverse().unwrap();
        let result=integrate(|x|{let z=&inverse*(x-&mean);-0.5*z.norm_squared()},&mean,&factor,3);
        let exact=3.*std::f64::consts::TAU.ln()+(0..6).map(|j|factor[(j,j)].ln()).sum::<f64>();
        assert!((result.log_value-exact).abs()<1e-11);
        assert!((&result.mean-&mean).norm()<1e-11);
        assert!((result.covariance-&factor*factor.transpose()).norm()<1e-11);
    }
    #[test]
    fn nonlinear_quadrature_converges() {
        let likelihood=|x:&DVector<f64>|-0.5*(x[0]-0.3).powi(2)-0.1*x[0].powi(4);
        let a=integrate(likelihood,&DVector::zeros(1),&DMatrix::identity(1,1),31);
        let b=integrate(likelihood,&DVector::zeros(1),&DMatrix::identity(1,1),61);
        assert!((a.log_value-b.log_value).abs()<1e-5);
        assert!((a.mean[0]-b.mean[0]).abs()<1e-5);
    }
}
