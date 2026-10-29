//! Cached per-pulse joint likelihood in path length and dechirped beat.
//! The change of coordinates retains delay-Doppler coupling; no extra data are added.
use super::*;
use nalgebra::{DMatrix,DVector};
use rand::{Rng,SeedableRng,rngs::StdRng};
use rand_distr::StandardNormal;
use std::time::Instant;

struct Surface {origin:[f64;2],half:[f64;2],n:usize,path_cells:usize,values:Vec<f64>,offset:f64}
pub struct Cache {pulses:Vec<super::reduced::Pulse>,offsets:Vec<f64>}

fn tune_scale(log_scale:f64,step:usize,accept:bool)->f64 {
    (log_scale+(step as f64+10.).powf(-0.6)*((if accept {1.}else{0.})-0.234)).clamp(-8.,4.)
}

struct RadiusTransport {reference:f64,beta:Vec<f64>}
impl RadiusTransport {
    fn new(x:&DVector<f64>,cov:&DMatrix<f64>)->Self {
        let beta=(0..6).map(|i|if x.len()==7 {cov[(i,6)]/cov[(6,6)]}else{0.}).collect();
        Self{reference:x[6],beta}
    }
    fn physical(&self,y:&DVector<f64>)->DVector<f64> {
        let mut x=y.clone();
        if y.len()!=7 || !(-7. ..=-2.).contains(&y[6]){return x;}
        let delta=y[6]-self.reference;
        // Drag varies as 1/r. Straighten this correlation while still sampling
        // log radius. This triangular shear has determinant one: no prior change.
        let shift=-(-std::f64::consts::LN_10*delta).exp_m1()/std::f64::consts::LN_10-delta;
        for i in 0..6 {x[i]+=self.beta[i]*shift;}
        x
    }
}

fn cubic(y:[f64;4],t:f64)->f64 {
    // Four-point Lagrange interpolation, exact for cubic polynomials.
    -t*(t-1.)*(t-2.)*y[0]/6.+(t+1.)*(t-1.)*(t-2.)*y[1]/2.
        -(t+1.)*t*(t-2.)*y[2]/2.+(t+1.)*t*(t-1.)*y[3]/6.
}
impl Surface {
    fn at(&self,x:[f64;2])->Option<f64> {
        let p=[(x[0]-self.origin[0])/self.half[0],(x[1]-self.origin[1])/self.half[1]];
        let z=p.map(|v|(v+1.)*0.5*(self.n-1) as f64);
        // Leave a one-cell interpolation guard; never extrapolate or clamp a posterior.
        if z.iter().any(|v| !v.is_finite() || *v<1. || *v>=(self.n-2) as f64){return None;}
        let a=z[0].floor() as usize;let b=z[1].floor() as usize;
        // Cell-averaged pulse support has derivative changes at sample edges.
        // Never fit a cubic across such an edge when the path grid is aligned.
        let start=if self.path_cells>=4 {
            let center=(self.n/2) as i64;let cells=self.path_cells as i64;
            let lo=(center+((a as i64-center).div_euclid(cells))*cells).max(0) as usize;
            let hi=(lo+self.path_cells).min(self.n-1);
            (a-1).clamp(lo,hi-3)
        }else{a-1};
        let mut row=[0.;4];
        for k in 0..4 {
            let values=std::array::from_fn(|j|self.values[(start+k)*self.n+b+j-1]);
            if values.iter().any(|v|!v.is_finite()){return None;}
            row[k]=cubic(values,z[1]-b as f64);
        }
        Some(cubic(row,z[0]-(start+1) as f64))
    }
}

impl Voltage {
    pub(super) fn pulse_logp(&self,i:usize,path:f64,beat:f64)->f64 {
        let fd=beat-self.bandwidth/self.duration*path/C;
        let fft=self.fft.as_ref().unwrap();let mut s=vec![Complex::new(0.,0.);fft.forward.len()];
        for j in 0..self.n {
            let sample=self.sample_time[i*self.n+j];let tau=sample-path/C;
            let env=((tau+0.5/self.fs)*self.fs).clamp(0.,1.)*((self.duration-tau+0.5/self.fs)*self.fs).clamp(0.,1.);
            let phase=std::f64::consts::TAU*(self.bandwidth/2.*tau-self.bandwidth/(2.*self.duration)*tau*tau+fd*sample);
            let (si,co)=phase.sin_cos();s[j]=Complex::new(env*co,env*si);
        }
        fft.forward.process(&mut s);
        for (z,f) in s.iter_mut().zip(&fft.filters[i]){*z*=f;}
        fft.inverse.process(&mut s);let scale=1./s.len() as f64;
        let mut cross=Complex::new(0.,0.);let mut norm=0.;
        for j in self.filter_len-1..self.n {
            let m=s[j]*scale;let k=i*self.n+j;
            cross+=m.conj()*Complex::new(self.white_real[k],self.white_imag[k]);norm+=m.norm_sqr();
        }
        if norm<10. {return f64::NEG_INFINITY;}
        // Data-only energy omitted. Same amplitude integral as voltage_sample.
        cross.norm_sqr()/(norm*self.noise[i])-norm.ln()
    }
}
impl Cache {
    fn build(v:&Voltage,q:&[f64],_n:usize,_expand:f64)->anyhow::Result<Self> {
        let obs=v.observables(q).ok_or_else(||anyhow::anyhow!("Invalid seed"))?;
        let mut pulses=Vec::new();let mut offsets=Vec::new();
        for (i,o) in obs.iter().enumerate() {
            let beat=o[1]+v.bandwidth/v.duration*o[0]/C;
            let p=super::reduced::Pulse::build(v,i,beat);
            let offset=v.pulse_logp(i,o[0],beat);
            anyhow::ensure!(offset.is_finite(),"Invalid pulse {i}");
            anyhow::ensure!((p.logp(o[0],beat)-offset).abs()<1e-5,"Reduced pulse {i} disagrees with raw voltage");
            pulses.push(p);offsets.push(offset);
        }
        Ok(Self{pulses,offsets})
    }
    /* Archived gridded construction, superseded by analytic envelope statistics.
    fn old_build(v:&Voltage,q:&[f64],n:usize,expand:f64)->anyhow::Result<Self> {
        anyhow::ensure!(n>=9 && n%2==1,"Odd grid size >=9 required");
        let obs=v.observables(q).ok_or_else(||anyhow::anyhow!("Invalid cache seed"))?;
        let k=v.bandwidth/v.duration/C;let mut surfaces=Vec::new();
        for (i,o) in obs.iter().enumerate() {
            let mut origin=[o[0],o[1]+k*o[0]];let offset=v.pulse_logp(i,origin[0],origin[1]);
            anyhow::ensure!(offset.is_finite(),"Invalid pulse {i}");
            let mut half=[0.;2];
            for j in 0..2 {
                let step=if j==0 {2.} else {5.};let mut a=origin;let mut b=origin;a[j]+=step;b[j]-=step;
                let curvature=(2.*offset-v.pulse_logp(i,a[0],a[1])-v.pulse_logp(i,b[0],b[1]))/(step*step);
                half[j]=expand*(12./curvature.max(1e-8).sqrt()).clamp(if j==0 {100.}else{150.},if j==0 {2500.}else{5000.});
            }
            let spacing=C/v.fs;let desired=2.*half[0]/(n-1) as f64;
            let exponent=(spacing/desired).log2().floor();
            let mut path_cells=0;
            if exponent>=2. && (v.duration*v.fs-(v.duration*v.fs).round()).abs()<1e-7 {
                path_cells=2usize.pow(exponent as u32);
                let edge=C*(v.sample_time[i*v.n]-0.5/v.fs);
                origin[0]=edge+((origin[0]-edge)/spacing).round()*spacing;
                half[0]=(n-1) as f64/2.*spacing/path_cells as f64;
            }
            let mut values=Vec::with_capacity(n*n);
            for a in 0..n {for b in 0..n {
                let x=origin[0]+half[0]*(2.*a as f64/(n-1) as f64-1.);
                let y=origin[1]+half[1]*(2.*b as f64/(n-1) as f64-1.);
                values.push(v.pulse_logp(i,x,y)-offset);
            }}
            surfaces.push(Surface{origin,half,n,path_cells,values,offset});
        }
        Ok(Self{surfaces})
    }
    */
    fn evaluate(&self,v:&Voltage,q:&[f64])->Option<Vec<f64>> {
        let obs=v.observables(q)?;let k=v.bandwidth/v.duration/C;
        Some(obs.iter().enumerate().map(|(i,o)|self.pulses[i].logp(o[0],o[1]+k*o[0])-self.offsets[i]).collect())
    }
    fn logp(&self,v:&Voltage,q:&[f64])->f64 {
        self.evaluate(v,q).map_or(f64::NEG_INFINITY,|x|x.iter().sum())
    }
    fn exact(&self,v:&Voltage,q:&[f64])->f64 {
        let Some(obs)=v.observables(q) else{return f64::NEG_INFINITY;};let k=v.bandwidth/v.duration/C;
        obs.iter().enumerate().map(|(i,o)|v.pulse_logp(i,o[0],o[1]+k*o[0])-self.offsets[i]).sum()
    }
    fn save(&self,path:&str)->anyhow::Result<()> {
        let h=hdf5::File::create(path)?;
        h.new_dataset_builder().with_data(&self.offsets).create("offset")?;
        h.new_dataset::<u64>().create("sufficient_statistics_version")?.write_scalar(&1)?;
        h.flush()?;Ok(())
    }
}

#[no_mangle] pub unsafe extern "C" fn voltage_cache_build(v:*const Voltage,q:*const f64,n:usize,expand:f64,path:*const c_char)->*mut Cache {
    let v=&*v;let result=(||->anyhow::Result<Cache>{let c=Cache::build(v,std::slice::from_raw_parts(q,v.dimension()),n,expand)?;c.save(CStr::from_ptr(path).to_str()?)?;Ok(c)})();
    match result {Ok(c)=>Box::into_raw(Box::new(c)),Err(e)=>{eprintln!("{e:#}");std::ptr::null_mut()}}
}
#[no_mangle] pub unsafe extern "C" fn voltage_cache_free(c:*mut Cache){if !c.is_null(){drop(Box::from_raw(c));}}
#[no_mangle] pub unsafe extern "C" fn voltage_cache_logp(c:*const Cache,v:*const Voltage,q:*const f64,exact:i32)->f64 {
    let v=&*v;let q=std::slice::from_raw_parts(q,v.dimension());if exact!=0 {(*c).exact(v,q)}else{(*c).logp(v,q)}
}
#[no_mangle] pub unsafe extern "C" fn voltage_cache_sample(c:*const Cache,v:*const Voltage,q:*const f64,cov:*const f64,
    warmup:usize,draws:usize,seconds:f64,seed:u64,path:*const c_char)->i32 {
    let result=(||->anyhow::Result<()> {
        let started=Instant::now();let v=&*v;let c=&*c;let n=v.dimension();
        let mut rng=StdRng::seed_from_u64(seed);let mut x=DVector::from_column_slice(std::slice::from_raw_parts(q,n));
        let base=DMatrix::from_row_slice(n,n,std::slice::from_raw_parts(cov,n*n));
        let transport=RadiusTransport::new(&x,&base);
        let mut physical_x=transport.physical(&x);
        let mut chol=base.clone().cholesky().ok_or_else(||anyhow::anyhow!("Proposal covariance not positive definite"))?.l()*(2.38/(n as f64).sqrt());
        let mut lp=c.logp(v,physical_x.as_slice());anyhow::ensure!(lp.is_finite(),"Nonfinite initial posterior");
        let mut mean=DVector::zeros(n);let mut scatter=DMatrix::zeros(n,n);
        let mut log_scale=0_f64;
        let mut samples=Vec::new();let mut logs=Vec::new();let mut accepted=0u64;let mut outside=0u64;let mut steps=0usize;
        for step in 0..warmup+draws {
            if started.elapsed().as_secs_f64()>=seconds {break;}
            let z=DVector::from_fn(n,|_,_|rng.sample::<f64,_>(StandardNormal));let proposal=&x+(&chol*z)*log_scale.exp();
            let physical_proposal=transport.physical(&proposal);
            let candidate=c.logp(v,physical_proposal.as_slice());if !candidate.is_finite(){outside+=1;}
            let accept=candidate.is_finite() && rng.gen::<f64>().ln()<candidate-lp;
            if accept {x=proposal;physical_x=physical_proposal;lp=candidate;if step>=warmup {accepted+=1;}}
            if step<warmup {
                // Robbins-Monro scale tuning is frozen with the covariance after warmup.
                log_scale=tune_scale(log_scale,step,accept);
                let count=(step+1) as f64;let delta=&x-&mean;mean+=&delta/count;scatter+=&delta*(&x-&mean).transpose();
                if step>=499 && step%100==99 {let covariance=(&scatter/(count-1.)+&base*0.01)*(2.38_f64.powi(2)/n as f64);if let Some(f)=covariance.cholesky(){chol=f.l();}}
            } else {samples.extend_from_slice(physical_x.as_slice());logs.push(lp);}
            steps=step+1;
        }
        let h=hdf5::File::create(CStr::from_ptr(path).to_str()?)?;
        h.new_dataset::<f64>().shape((logs.len(),n)).create("samples")?.write_raw(&samples)?;
        h.new_dataset_builder().with_data(&logs).create("log_probability")?;
        for (name,value) in [("accepted",accepted),("steps",steps as u64),("outside_proposals",outside)]{h.new_dataset::<u64>().create(name)?.write_scalar(&value)?;}
        h.new_dataset::<f64>().create("elapsed_seconds")?.write_scalar(&started.elapsed().as_secs_f64())?;
        h.new_dataset::<f64>().create("frozen_proposal_scale")?.write_scalar(&log_scale.exp())?;
        h.new_dataset_builder().with_data(&transport.beta).create("radius_transport_beta")?;
        h.new_dataset::<f64>().create("radius_transport_reference")?.write_scalar(&transport.reference)?;
        h.flush()?;Ok(())
    })();match result {Ok(())=>1,Err(e)=>{eprintln!("{e:#}");0}}
}

#[no_mangle] pub unsafe extern "C" fn voltage_cache_margin(c:*const Cache,v:*const Voltage,q:*const f64)->f64 {
    let v=&*v;let c=&*c;let q=std::slice::from_raw_parts(q,v.dimension());
    if c.logp(v,q).is_finite(){1.}else{-1.}
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn inverse_radius_transport_preserves_volume_and_local_metric() {
        let mut center=DVector::zeros(7);center[6]=-4.;
        let mut cov=DMatrix::identity(7,7);cov[(0,6)]=0.7;cov[(6,0)]=0.7;
        let t=RadiusTransport::new(&center,&cov);
        for radius in [-6.,-4.,-2.1] {
            let mut y=center.clone();y[6]=radius;
            let mut jac=DMatrix::zeros(7,7);
            for j in 0..7 {let mut a=y.clone();let mut b=y.clone();a[j]+=1e-5;b[j]-=1e-5;
                let column=(t.physical(&a)-t.physical(&b))/(2e-5);jac.set_column(j,&column);}
            assert!((jac.determinant()-1.).abs()<1e-7);
            if radius==-4. {assert!((jac-DMatrix::identity(7,7)).norm()<1e-8);}
        }
    }
    #[test] fn warmup_scale_recovers_gaussian_from_oversized_steps() {
        let mut rng=StdRng::seed_from_u64(515);
        let mut x=[0_f64;7];let mut lp=0.;let mut scale=0.;
        let mut sum=[0.;7];let mut square=[0.;7];let mut accepted=0;
        for step in 0..104000 {
            let y: [f64;7]=std::array::from_fn(|i|x[i]+50.*f64::exp(scale)*rng.sample::<f64,_>(StandardNormal));
            let candidate=-0.5*y.iter().map(|v|v*v).sum::<f64>();
            let accept=rng.gen::<f64>().ln()<candidate-lp;
            if accept {x=y;lp=candidate;}
            if step<4000 {scale=tune_scale(scale,step,accept);}else {
                if accept {accepted+=1;}
                for i in 0..7 {sum[i]+=x[i];square[i]+=x[i]*x[i];}
            }
        }
        assert!((10000..50000).contains(&accepted));
        for i in 0..7 {assert!((sum[i]/100000.).abs()<0.05);assert!((square[i]/100000.-1.).abs()<0.08);}
    }
    #[test] fn cubic_recovers_polynomial() {
        let f=|x:f64|2.+3.*x-4.*x*x+0.7*x*x*x;
        for i in 0..100 {let t=i as f64/100.;assert!((cubic([-1.,0.,1.,2.].map(f),t)-f(t)).abs()<1e-12);}
    }
    #[test] fn interpolation_respects_sample_edge_cusp() {
        let n=65;let mut values=Vec::new();
        for i in 0..n {for j in 0..n {let x=i as f64-32.;let y=j as f64-32.;values.push(-x.abs()-y*y);}}
        let surface=Surface{origin:[0.,0.],half:[32.,32.],n,path_cells:16,values,offset:0.};
        for i in -10..10 {let x=i as f64/11.;let y=0.37;assert!((surface.at([x,y]).unwrap()+x.abs()+y*y).abs()<1e-10);}
    }
}
