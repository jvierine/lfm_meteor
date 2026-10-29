//! Experimental sampled-voltage likelihood. No calibrated receiver filter assumed.
use std::ffi::{c_char,CStr};
use nalgebra::{Matrix3,Vector3};
use crate::{C,metropolis};

pub struct Voltage {
    center:Vec<f64>, time:Vec<f64>, frames:Vec<f64>, rx:Vec<f64>, tx:Vec<f64>,
    sample_time:Vec<f64>, real:Vec<f64>, imag:Vec<f64>, noise:Vec<f64>,
    whiten_real:Vec<f64>,whiten_imag:Vec<f64>,white_real:Vec<f64>,white_imag:Vec<f64>,filter_len:usize,
    n:usize, fs:f64, bandwidth:f64, duration:f64, wavelength:f64,
}
impl Voltage {
    fn load(path:&str)->anyhow::Result<Self> {
        let h=hdf5::File::open(path)?;
        let r=|key:&str|->anyhow::Result<Vec<f64>> {Ok(h.dataset(key)?.read_raw()?)};
        let time=r("time_s")?; let real=r("real")?;
        anyhow::ensure!(!time.is_empty() && real.len()%time.len()==0,"Invalid voltage shape");
        let n=real.len()/time.len();
        let mut v=Self {center:r("center")?,frames:r("frames")?,rx:r("rx")?,tx:r("tx")?,
            sample_time:r("sample_time_s")?,imag:r("imag")?,noise:r("noise_power")?,
            fs:h.dataset("fs")?.read_scalar()?,bandwidth:h.dataset("bandwidth")?.read_scalar()?,
            duration:h.dataset("duration")?.read_scalar()?,wavelength:h.dataset("wavelength")?.read_scalar()?,
            whiten_real:r("whitener_real")?,whiten_imag:r("whitener_imag")?,white_real:Vec::new(),white_imag:Vec::new(),filter_len:17,
            time,real,n};
        anyhow::ensure!(v.center.len()==6 && v.frames.len()==42*v.time.len() &&
            v.rx.len()==3*v.time.len() && v.tx.len()==3 && v.sample_time.len()==v.real.len() &&
            v.imag.len()==v.real.len() && v.noise.len()==v.time.len() &&
            v.whiten_real.len()==17*v.time.len() && v.whiten_imag.len()==v.whiten_real.len(),"Invalid prepared arrays");
        v.white_real=vec![0.;v.real.len()];v.white_imag=vec![0.;v.imag.len()];
        for i in 0..v.time.len() {
            for j in v.filter_len-1..v.n {
                for lag in 0..v.filter_len {
                    let k=i*v.n+j-lag;let f=i*v.filter_len+lag;
                    v.white_real[i*v.n+j]+=v.whiten_real[f]*v.real[k]-v.whiten_imag[f]*v.imag[k];
                    v.white_imag[i*v.n+j]+=v.whiten_real[f]*v.imag[k]+v.whiten_imag[f]*v.real[k];
                }
            }
        }
        Ok(v)
    }
    fn frame(&self,i:usize,x:Vector3<f64>,v:Vector3<f64>)->(Vector3<f64>,Vector3<f64>) {
        let f=&self.frames[i*42..(i+1)*42]; let m=|s|Matrix3::from_row_slice(&f[s..s+9]);
        (Vector3::from_column_slice(&f[..3])+m(6)*x+m(24)*v,
         Vector3::from_column_slice(&f[3..6])+m(15)*x+m(33)*v)
    }
    pub fn residual(&self,q:&[f64],out:&mut[f64],log_norm:&mut f64)->bool {
        self.evaluate(q,out,log_norm,false)
    }
    fn evaluate(&self,q:&[f64],out:&mut[f64],log_norm:&mut f64,prediction:bool)->bool {
        *log_norm=0.;
        if q.len()!=8 || q.iter().any(|x|!x.is_finite()) || q[..3].iter().any(|x|x.abs()>20.) ||
            q[3..6].iter().any(|x|x.abs()>40.) || !(-5. ..=5.).contains(&q[6]) ||
            !(-4. ..=3.).contains(&q[7]) {return false;}
        let x0=Vector3::from_fn(|j,_|self.center[j]+1000.*q[j]);
        let v0=Vector3::from_fn(|j,_|self.center[3+j]+1000.*q[3+j]);
        let speed=v0.norm(); if !(1000. ..150000.).contains(&speed) {return false;}
        let u=v0/speed; let a=10_f64.powf(q[6]); let b=10_f64.powf(q[7]);
        let tx=Vector3::from_column_slice(&self.tx);
        let mut model=vec![(0.,0.);self.n];
        let mut white_model=vec![(0.,0.);self.n];
        for (i,&t) in self.time.iter().enumerate() {
            let e=(b*t).exp(); let v=speed-a*e;
            if !v.is_finite() || v<=100. {return false;}
            let s=speed*t-a*(b*t).exp_m1()/b;
            let nominal=x0+u*s; let vel=u*v; let mut x=nominal;
            for _ in 0..2 {let (p,_)=self.frame(i,x,vel);x=nominal-vel*((p-tx).norm()/C);}
            let (p,w)=self.frame(i,x,vel); let rx=Vector3::from_column_slice(&self.rx[i*3..i*3+3]);
            let ltx=p-tx; let lrx=p-rx;
            let path=ltx.norm()+lrx.norm(); let fd=-(ltx.normalize()+lrx.normalize()).dot(&w)/self.wavelength;
            let mut ar=0.;let mut ai=0.;let mut norm=0.;
            for (j,m) in model.iter_mut().enumerate() {
                let k=i*self.n+j;let sample=self.sample_time[k];let tau=sample-path/C;
                // Cell-averaged rectangular support avoids discontinuous sample inclusion.
                // Receiver filtering and intra-pulse motion are not yet calibrated.
                let envelope=((tau+0.5/self.fs)*self.fs).clamp(0.,1.) *
                    ((self.duration-tau+0.5/self.fs)*self.fs).clamp(0.,1.);
                let phase=std::f64::consts::TAU*(self.bandwidth/2.*tau-
                    self.bandwidth/(2.*self.duration)*tau*tau+fd*sample);
                let (si,co)=phase.sin_cos();*m=(envelope*co,envelope*si);
            }
            // Condition on off-echo initial samples, whose distribution is independent
            // of trajectory. Apply the same complex FIR to model and measured voltage.
            if model[..self.filter_len-1].iter().any(|m|m.0*m.0+m.1*m.1>1e-20){return false;}
            for j in self.filter_len-1..self.n {
                let mut wr=0.;let mut wi=0.;
                for lag in 0..self.filter_len {
                    let m=model[j-lag];let f=i*self.filter_len+lag;
                    wr+=self.whiten_real[f]*m.0-self.whiten_imag[f]*m.1;
                    wi+=self.whiten_real[f]*m.1+self.whiten_imag[f]*m.0;
                }
                white_model[j]=(wr,wi);let k=i*self.n+j;
                ar+=wr*self.white_real[k]+wi*self.white_imag[k];
                ai+=wr*self.white_imag[k]-wi*self.white_real[k];norm+=wr*wr+wi*wi;
            }
            if norm<10. {return false;}
            *log_norm+=norm.ln();
            ar/=norm; ai/=norm; let sigma=(self.noise[i]/2.).sqrt();
            for (j,m) in model.iter().enumerate() {
                let k=i*self.n+j;
                if prediction {
                    out[2*k]=ar*m.0-ai*m.1;out[2*k+1]=ar*m.1+ai*m.0;
                } else if j<self.filter_len-1 {out[2*k]=0.;out[2*k+1]=0.;}
                else {
                    let m=white_model[j];
                    out[2*k]=(ar*m.0-ai*m.1-self.white_real[k])/sigma;
                    out[2*k+1]=(ar*m.1+ai*m.0-self.white_imag[k])/sigma;
                }
            }
        }
        true
    }
}

// Python is used only for preparation, optimizer orchestration and plots;
// waveform evaluations and Metropolis-Hastings sampling run in Rust.
#[no_mangle] pub unsafe extern "C" fn voltage_load(path:*const c_char)->*mut Voltage {
    match Voltage::load(CStr::from_ptr(path).to_str().unwrap_or("")) {
        Ok(v)=>Box::into_raw(Box::new(v)), Err(e)=>{eprintln!("{e:#}");std::ptr::null_mut()}
    }
}
#[no_mangle] pub unsafe extern "C" fn voltage_len(v:*const Voltage)->usize {2*(*v).real.len()}
#[no_mangle] pub unsafe extern "C" fn voltage_residual(v:*const Voltage,q:*const f64,out:*mut f64)->i32 {
    let v=&*v;
    v.residual(std::slice::from_raw_parts(q,8),std::slice::from_raw_parts_mut(out,2*v.real.len()),&mut 0.) as i32
}
#[no_mangle] pub unsafe extern "C" fn voltage_free(v:*mut Voltage) {if !v.is_null(){drop(Box::from_raw(v));}}
#[no_mangle] pub unsafe extern "C" fn voltage_prediction(v:*const Voltage,q:*const f64,out:*mut f64)->i32 {
    let v=&*v;
    v.evaluate(std::slice::from_raw_parts(q,8),std::slice::from_raw_parts_mut(out,2*v.real.len()),&mut 0.,true) as i32
}
#[no_mangle] pub unsafe extern "C" fn voltage_sample(v:*const Voltage,q:*const f64,widths:*const f64,
    warmup:usize,draws:usize,seed:u64,path:*const c_char)->i32 {
    let v=&*v;
    let logp=|q:&[f64]|{let mut r=vec![0.;2*v.real.len()];let mut log_norm=0.;
        // Integrate each complex amplitude with a flat Re/Im prior, rather than
        // treating its profiled estimate as known during posterior sampling.
        if v.residual(q,&mut r,&mut log_norm){-0.5*r.iter().map(|x|x*x).sum::<f64>()-log_norm}else{f64::NEG_INFINITY}};
    let result=(||->anyhow::Result<()> {
        let chain=metropolis(logp,std::slice::from_raw_parts(q,8),std::slice::from_raw_parts(widths,8),warmup,draws,seed)?;
        let h=hdf5::File::create(CStr::from_ptr(path).to_str()?)?;
        let flat:Vec<f64>=chain.samples.iter().flatten().copied().collect();
        h.new_dataset::<f64>().shape((draws,8)).create("samples")?.write_raw(&flat)?;
        h.new_dataset_builder().with_data(&chain.log_probability).create("log_probability")?;
        h.new_dataset::<u64>().create("accepted")?.write_scalar(&(chain.accepted as u64))?;
        h.flush()?;Ok(())
    })();
    match result {Ok(())=>1,Err(e)=>{eprintln!("{e:#}");0}}
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn noiseless_waveform_and_delay_perturbation() {
        let n=1000;let fs=4e6;let duration=200e-6;let bandwidth=4e6;let wavelength=0.7;
        let beta=9900./C;let path=200000.*(1.+beta+beta*beta);let fd=19800./wavelength;
        let mut frame=vec![0.;42];
        for j in 0..3 {frame[6+3*j+j]=1.;frame[33+3*j+j]=1.;}
        let sample_time:Vec<f64>=(0..n).map(|j|path/C+(j as f64-100.)/fs).collect();
        let mut real=Vec::new();let mut imag=Vec::new();
        for &sample in &sample_time {
            let t=sample-path/C;
            let env=((t+0.5/fs)*fs).clamp(0.,1.)*((duration-t+0.5/fs)*fs).clamp(0.,1.);
            let angle=std::f64::consts::TAU*(bandwidth/2.*t-bandwidth/(2.*duration)*t*t+fd*sample);
            real.push(env*(2.*angle.cos()-angle.sin()));imag.push(env*(2.*angle.sin()+angle.cos()));
        }
        let model=Voltage {center:vec![0.,0.,100000.,0.,0.,-10000.],time:vec![0.],frames:frame,
            rx:vec![0.;3],tx:vec![0.;3],sample_time,white_real:real.clone(),white_imag:imag.clone(),
            whiten_real:vec![1.],whiten_imag:vec![0.],filter_len:1,real,imag,noise:vec![1.],n,fs,bandwidth,duration,wavelength};
        let mut q=vec![0.,0.,0.,0.,0.,0.,2.,0.];let mut residual=vec![0.;2*n];let mut lognorm=0.;
        assert!(model.residual(&q,&mut residual,&mut lognorm));
        assert!(residual.iter().map(|r|r*r).sum::<f64>()<1e-12);
        q[2]=0.001;
        assert!(model.residual(&q,&mut residual,&mut lognorm));
        assert!(residual.iter().map(|r|r*r).sum::<f64>()>0.001);
    }
}
