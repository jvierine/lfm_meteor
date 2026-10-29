//! Cached per-pulse joint likelihood in path length and dechirped beat.
//! The change of coordinates retains delay-Doppler coupling; no extra data are added.
use super::*;
use nalgebra::{DMatrix,DVector};
use rand::{Rng,SeedableRng,rngs::StdRng};
use rand_distr::StandardNormal;
use std::time::Instant;

struct Surface {origin:[f64;2],half:[f64;2],n:usize,values:Vec<f64>,offset:f64}
pub struct Cache {surfaces:Vec<Surface>}

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
        let mut row=[0.;4];
        for k in 0..4 {
            let values=std::array::from_fn(|j|self.values[(a+k-1)*self.n+b+j-1]);
            if values.iter().any(|v|!v.is_finite()){return None;}
            row[k]=cubic(values,z[1]-b as f64);
        }
        Some(cubic(row,z[0]-a as f64))
    }
}

impl Voltage {
    fn pulse_logp(&self,i:usize,path:f64,beat:f64)->f64 {
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
    fn build(v:&Voltage,q:&[f64],n:usize,expand:f64)->anyhow::Result<Self> {
        anyhow::ensure!(n>=9 && n%2==1,"Odd grid size >=9 required");
        let obs=v.observables(q).ok_or_else(||anyhow::anyhow!("Invalid cache seed"))?;
        let k=v.bandwidth/v.duration/C;let mut surfaces=Vec::new();
        for (i,o) in obs.iter().enumerate() {
            let origin=[o[0],o[1]+k*o[0]];let offset=v.pulse_logp(i,origin[0],origin[1]);
            anyhow::ensure!(offset.is_finite(),"Invalid pulse {i}");
            let mut half=[0.;2];
            for j in 0..2 {
                let step=if j==0 {2.} else {5.};let mut a=origin;let mut b=origin;a[j]+=step;b[j]-=step;
                let curvature=(2.*offset-v.pulse_logp(i,a[0],a[1])-v.pulse_logp(i,b[0],b[1]))/(step*step);
                half[j]=expand*(12./curvature.max(1e-8).sqrt()).clamp(if j==0 {100.}else{150.},if j==0 {2500.}else{5000.});
            }
            let mut values=Vec::with_capacity(n*n);
            for a in 0..n {for b in 0..n {
                let x=origin[0]+half[0]*(2.*a as f64/(n-1) as f64-1.);
                let y=origin[1]+half[1]*(2.*b as f64/(n-1) as f64-1.);
                values.push(v.pulse_logp(i,x,y)-offset);
            }}
            surfaces.push(Surface{origin,half,n,values,offset});
        }
        Ok(Self{surfaces})
    }
    fn evaluate(&self,v:&Voltage,q:&[f64])->Option<Vec<f64>> {
        let obs=v.observables(q)?;let k=v.bandwidth/v.duration/C;
        obs.iter().enumerate().map(|(i,o)|self.surfaces[i].at([o[0],o[1]+k*o[0]])).collect()
    }
    fn logp(&self,v:&Voltage,q:&[f64])->f64 {
        self.evaluate(v,q).map_or(f64::NEG_INFINITY,|x|x.iter().sum())
    }
    fn exact(&self,v:&Voltage,q:&[f64])->f64 {
        let Some(obs)=v.observables(q) else{return f64::NEG_INFINITY;};let k=v.bandwidth/v.duration/C;
        obs.iter().enumerate().map(|(i,o)|v.pulse_logp(i,o[0],o[1]+k*o[0])-self.surfaces[i].offset).sum()
    }
    fn save(&self,path:&str)->anyhow::Result<()> {
        let h=hdf5::File::create(path)?;let n=self.surfaces[0].n;
        let origin:Vec<f64>=self.surfaces.iter().flat_map(|s|s.origin).collect();
        let half:Vec<f64>=self.surfaces.iter().flat_map(|s|s.half).collect();
        let values:Vec<f64>=self.surfaces.iter().flat_map(|s|s.values.clone()).collect();
        h.new_dataset::<f64>().shape((self.surfaces.len(),2)).create("origin_path_m_beat_hz")?.write_raw(&origin)?;
        h.new_dataset::<f64>().shape((self.surfaces.len(),2)).create("half_width")?.write_raw(&half)?;
        h.new_dataset::<f64>().shape((self.surfaces.len(),n,n)).create("relative_log_likelihood")?.write_raw(&values)?;
        h.new_dataset_builder().with_data(&self.surfaces.iter().map(|s|s.offset).collect::<Vec<_>>()).create("offset")?;
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
        let mut chol=base.clone().cholesky().ok_or_else(||anyhow::anyhow!("Proposal covariance not positive definite"))?.l()*(2.38/(n as f64).sqrt());
        let mut lp=c.logp(v,x.as_slice());anyhow::ensure!(lp.is_finite(),"Nonfinite initial posterior");
        let mut mean=DVector::zeros(n);let mut scatter=DMatrix::zeros(n,n);
        let mut samples=Vec::new();let mut logs=Vec::new();let mut accepted=0u64;let mut outside=0u64;let mut steps=0usize;
        for step in 0..warmup+draws {
            if started.elapsed().as_secs_f64()>=seconds {break;}
            let z=DVector::from_fn(n,|_,_|rng.sample::<f64,_>(StandardNormal));let proposal=&x+&chol*z;
            let candidate=c.logp(v,proposal.as_slice());if !candidate.is_finite(){outside+=1;}
            if candidate.is_finite() && rng.gen::<f64>().ln()<candidate-lp {x=proposal;lp=candidate;if step>=warmup {accepted+=1;}}
            if step<warmup {
                let count=(step+1) as f64;let delta=&x-&mean;mean+=&delta/count;scatter+=&delta*(&x-&mean).transpose();
                if step>=499 && step%100==99 {let covariance=(&scatter/(count-1.)+&base*0.01)*(2.38_f64.powi(2)/n as f64);if let Some(f)=covariance.cholesky(){chol=f.l();}}
            } else {samples.extend_from_slice(x.as_slice());logs.push(lp);}
            steps=step+1;
        }
        let h=hdf5::File::create(CStr::from_ptr(path).to_str()?)?;
        h.new_dataset::<f64>().shape((logs.len(),n)).create("samples")?.write_raw(&samples)?;
        h.new_dataset_builder().with_data(&logs).create("log_probability")?;
        for (name,value) in [("accepted",accepted),("steps",steps as u64),("outside_proposals",outside)]{h.new_dataset::<u64>().create(name)?.write_scalar(&value)?;}
        h.new_dataset::<f64>().create("elapsed_seconds")?.write_scalar(&started.elapsed().as_secs_f64())?;
        h.flush()?;Ok(())
    })();match result {Ok(())=>1,Err(e)=>{eprintln!("{e:#}");0}}
}

#[no_mangle] pub unsafe extern "C" fn voltage_cache_margin(c:*const Cache,v:*const Voltage,q:*const f64)->f64 {
    let v=&*v;let c=&*c;let q=std::slice::from_raw_parts(q,v.dimension());
    let Some(obs)=v.observables(q) else{return -1.;};let k=v.bandwidth/v.duration/C;
    obs.iter().enumerate().flat_map(|(i,o)|{
        let s=&c.surfaces[i];let b=o[1]+k*o[0];
        [1.-((o[0]-s.origin[0])/s.half[0]).abs(),1.-((b-s.origin[1])/s.half[1]).abs()]
    }).fold(1.,f64::min)
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn cubic_recovers_polynomial() {
        let f=|x:f64|2.+3.*x-4.*x*x+0.7*x*x*x;
        for i in 0..100 {let t=i as f64/100.;assert!((cubic([-1.,0.,1.,2.].map(f),t)-f(t)).abs()<1e-12);}
    }
}
