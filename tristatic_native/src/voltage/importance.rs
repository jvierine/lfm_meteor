//! Independent defensive importance sampling of exactly seven physical parameters.
use super::*;
use super::compressed::{Cache, RadiusTransport};
use nalgebra::{DMatrix,DVector};
use rand::{Rng,SeedableRng,rngs::StdRng};
use rand_distr::StandardNormal;
use std::time::Instant;

struct Proposal {
    center:DVector<f64>,factor:DMatrix<f64>,inverse:DMatrix<f64>,conditional:DMatrix<f64>,
    conditional_inverse:DMatrix<f64>,beta:DVector<f64>,logdet:f64,conditional_logdet:f64,
}

fn student_constant(dimension:usize)->f64 {
    let gamma_ratio=if dimension==6 {120_f64.ln()}else{
        0.5*std::f64::consts::PI.ln()+(0..7).map(|j|(j as f64+0.5).ln()).sum::<f64>()-6_f64.ln()
    };
    gamma_ratio-0.5*dimension as f64*(8.*std::f64::consts::PI).ln()
}

fn newton_mode<F:Fn(&DVector<f64>)->f64>(logp:F,initial:DVector<f64>,factor:&DMatrix<f64>,seconds:f64)->DVector<f64> {
    let began=Instant::now();let mut x=initial;let n=x.len();
    for _ in 0..50 {
        if began.elapsed().as_secs_f64()>seconds {break;}
        let origin=logp(&x);if !origin.is_finite(){break;}
        let mut steps=vec![0.08_f64;n];let mut first=vec![0.;n];let mut second=vec![0.;n];
        let mut valid=true;
        for j in 0..n {
            if n==7 && factor[(6,j)]!=0. {
                let sign=if factor[(6,j)]*( -4.5-x[6])>=0. {1.}else{-1.};steps[j]*=sign;
            }
            let mut found=false;
            for _ in 0..12 {
                first[j]=logp(&(&x+factor.column(j)*steps[j]));
                second[j]=logp(&(&x+factor.column(j)*(2.*steps[j])));
                if first[j].is_finite() && second[j].is_finite(){found=true;break;}
                steps[j]*=0.5;
            }
            if !found {valid=false;break;}
        }
        if !valid {break;}
        let mut gradient=DVector::zeros(n);let mut information=DMatrix::zeros(n,n);
        for j in 0..n {
            gradient[j]=(-3.*origin+4.*first[j]-second[j])/(2.*steps[j]);
            information[(j,j)]=-(second[j]-2.*first[j]+origin)/steps[j].powi(2);
            for k in 0..j {
                let corner=logp(&(&x+factor.column(j)*steps[j]+factor.column(k)*steps[k]));
                let value=if corner.is_finite(){-(corner-first[j]-first[k]+origin)/(steps[j]*steps[k])}else{0.};
                information[(j,k)]=value;information[(k,j)]=value;
            }
        }
        if gradient.norm()<1e-3 {break;}
        let eig=information.symmetric_eigen();let floor=(eig.eigenvalues.max()*1e-4).max(0.05);
        let mut direction=&eig.eigenvectors*DMatrix::from_diagonal(&eig.eigenvalues.map(|x|1./x.max(floor)))
            *eig.eigenvectors.transpose()*gradient;
        let norm=direction.norm();if norm>30. {direction*=30./norm;}
        let mut changed=false;
        for k in 0..16 {
            let candidate=&x+factor*&direction*2_f64.powi(-k);
            if logp(&candidate)>origin+1e-8 {x=candidate;changed=true;break;}
        }
        if !changed {break;}
    }
    x
}

#[no_mangle]
pub unsafe extern "C" fn voltage_importance_optimize(cache:*const Cache,voltage:*const Voltage,
    initial:*const f64,covariance:*const f64,seconds:f64,result:*mut f64)->i32 {
    let cache=&*cache;let voltage=&*voltage;
    let initial=DVector::from_column_slice(std::slice::from_raw_parts(initial,7));
    let covariance=DMatrix::from_row_slice(7,7,std::slice::from_raw_parts(covariance,49));
    let Some(factor)=covariance.cholesky() else{return 0;};
    let optimized=newton_mode(|q|cache.logp(voltage,q.as_slice()),initial,&factor.l(),seconds);
    std::slice::from_raw_parts_mut(result,7).copy_from_slice(optimized.as_slice());
    if cache.logp(voltage,optimized.as_slice()).is_finite(){1}else{0}
}

impl Proposal {
    fn new(center:DVector<f64>,covariance:DMatrix<f64>)->Option<Self> {
        let factor=covariance.clone().cholesky()?.l();let inverse=factor.clone().try_inverse()?;
        let permutation=[6,0,1,2,3,4,5];
        let reordered=DMatrix::from_fn(7,7,|i,j|covariance[(permutation[i],permutation[j])]);
        let conditional=reordered.cholesky()?.l().view((1,1),(6,6)).into_owned();
        let conditional_inverse=conditional.clone().try_inverse()?;
        let beta=covariance.column(6).rows(0,6).into_owned()/covariance[(6,6)];
        let logdet=(0..7).map(|j|factor[(j,j)].ln()).sum();
        let conditional_logdet=(0..6).map(|j|conditional[(j,j)].ln()).sum();
        Some(Self{center,factor,inverse,conditional,conditional_inverse,beta,logdet,conditional_logdet})
    }
    fn draw(&self,rng:&mut StdRng)->DVector<f64> {
        let component=rng.gen::<f64>();
        let normal=DVector::from_fn(7,|_,_|rng.sample::<f64,_>(StandardNormal));
        if component<0.85 {return &self.center+&self.factor*normal;}
        let chi:f64=(0..8).map(|_|rng.sample::<f64,_>(StandardNormal).powi(2)).sum();
        let scale=2./(chi/8.).sqrt();
        if component<0.95 {return &self.center+(&self.factor*normal)*scale;}
        let radius=rng.gen_range(-7_f64..-2_f64);let mut x=self.center.clone();
        let mean=self.center.rows(0,6).into_owned()+&self.beta*(radius-self.center[6]);
        let first=mean+(&self.conditional*normal.rows(0,6))*scale;
        x.rows_mut(0,6).copy_from(&first);x[6]=radius;x
    }
    fn logp(&self,x:&DVector<f64>)->f64 {
        let z=&self.inverse*(x-&self.center);let squared=z.norm_squared();
        let gaussian=0.85_f64.ln()-3.5*std::f64::consts::TAU.ln()-self.logdet-0.5*squared;
        let student=0.10_f64.ln()+student_constant(7)-self.logdet-7.*2_f64.ln()-7.5*(1.+squared/32.).ln();
        let global=if (-7. ..=-2.).contains(&x[6]) {
            let mean=self.center.rows(0,6).into_owned()+&self.beta*(x[6]-self.center[6]);
            let z=&self.conditional_inverse*(x.rows(0,6).into_owned()-mean);
            0.05_f64.ln()-5_f64.ln()+student_constant(6)-self.conditional_logdet-6.*2_f64.ln()-7.*(1.+z.norm_squared()/32.).ln()
        }else{f64::NEG_INFINITY};
        let maximum=gaussian.max(student).max(global);
        maximum+((gaussian-maximum).exp()+(student-maximum).exp()+(global-maximum).exp()).ln()
    }
}

#[no_mangle]
pub unsafe extern "C" fn voltage_importance_sample(cache:*const Cache,voltage:*const Voltage,
    center:*const f64,covariance:*const f64,draws:usize,seed:u64,path:*const c_char)->i32 {
    let result=(||->anyhow::Result<()> {
        let began=Instant::now();let cache=&*cache;let voltage=&*voltage;
        let center=DVector::from_column_slice(std::slice::from_raw_parts(center,7));
        let covariance=DMatrix::from_row_slice(7,7,std::slice::from_raw_parts(covariance,49));
        let transport=RadiusTransport::new(&center,&covariance);
        let proposal=Proposal::new(center,covariance).ok_or_else(||anyhow::anyhow!("Invalid importance proposal"))?;
        let mut rng=StdRng::seed_from_u64(seed);let mut samples=Vec::with_capacity(7*draws);
        let mut log_weights=Vec::with_capacity(draws);let mut log_likelihood=Vec::with_capacity(draws);
        for _ in 0..draws {
            let coordinates=proposal.draw(&mut rng);let physical=transport.physical(&coordinates);
            let likelihood=cache.logp(voltage,physical.as_slice());
            log_weights.push(likelihood-proposal.logp(&coordinates));log_likelihood.push(likelihood);
            samples.extend_from_slice(physical.as_slice());
        }
        let h=hdf5::File::create(CStr::from_ptr(path).to_str()?)?;
        h.new_dataset::<f64>().shape((draws,7)).create("samples")?.write_raw(&samples)?;
        h.new_dataset_builder().with_data(&log_weights).create("log_weights")?;
        h.new_dataset_builder().with_data(&log_likelihood).create("log_likelihood")?;
        h.new_dataset::<f64>().create("elapsed_seconds")?.write_scalar(&began.elapsed().as_secs_f64())?;
        h.flush()?;Ok(())
    })();match result {Ok(())=>1,Err(e)=>{eprintln!("{e:#}");0}}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn newton_initializer_recovers_from_perturbed_correlated_start() {
        let truth=DVector::from_vec(vec![0.2,-0.3,0.1,0.4,0.2,-0.1,-4.]);
        let mut factor=DMatrix::identity(7,7)*0.02;for j in 1..7 {factor[(j,j-1)]=0.008;}
        let inverse=factor.clone().try_inverse().unwrap();
        let initial=&truth+&factor*DVector::from_vec(vec![8.,-12.,6.,-4.,10.,-7.,9.]);
        let objective=|x:&DVector<f64>|{let z=&inverse*(x-&truth);-0.5*z.norm_squared()-0.01*z[6].powi(4)};
        let result=newton_mode(objective,initial,&factor,2.);
        assert!((&inverse*(result-truth)).norm()<0.002);
    }
    #[test]
    fn independent_weights_recover_correlated_seven_dimensional_density() {
        let center=DVector::from_vec(vec![0.2,-0.3,0.1,0.4,0.2,-0.1,-4.]);
        let mut factor=DMatrix::identity(7,7)*0.2;
        for j in 1..7 {factor[(j,j-1)]=0.08;}
        factor[(6,6)]=0.03;
        let inverse=factor.clone().try_inverse().unwrap();
        let proposal=Proposal::new(center.clone(),&factor*factor.transpose()).unwrap();
        let mut rng=StdRng::seed_from_u64(618);
        let mut total=0.;let mut first=DVector::zeros(7);let mut second=DMatrix::zeros(7,7);let n=80000;
        for _ in 0..n {
            let x=proposal.draw(&mut rng);
            let z=&inverse*(&x-&center);
            let weight=if (-7. ..=-2.).contains(&x[6]) {
                (-0.5*z.norm_squared()-proposal.logp(&x)-3.5*std::f64::consts::TAU.ln()
                    -(0..7).map(|j|factor[(j,j)].ln()).sum::<f64>()).exp()
            }else{0.};
            total+=weight;first+=&x*weight;second+=(&x*x.transpose())*weight;
        }
        let mean=first/total;let covariance=second/total-&mean*mean.transpose();
        assert!((total/n as f64-1.).abs()<0.01);
        assert!((&mean-&center).norm()<0.008);
        assert!((covariance-&factor*factor.transpose()).norm()<0.004);
    }
}
