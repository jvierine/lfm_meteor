//! Experimental sampled-voltage likelihood. No calibrated receiver filter assumed.
use std::ffi::{c_char,CStr};
use nalgebra::{Matrix3,Vector3};
use crate::{C,metropolis};
use rustfft::{Fft,FftPlanner,num_complex::Complex};
use std::sync::Arc;
mod compressed;
mod reduced;

struct WhiteningFft {forward:Arc<dyn Fft<f64>>,inverse:Arc<dyn Fft<f64>>,filters:Vec<Vec<Complex<f64>>>}

pub struct Voltage {
    center:Vec<f64>, time:Vec<f64>, frames:Vec<f64>, rx:Vec<f64>, tx:Vec<f64>,
    sample_time:Vec<f64>, real:Vec<f64>, imag:Vec<f64>, noise:Vec<f64>,
    whiten_real:Vec<f64>,whiten_imag:Vec<f64>,white_real:Vec<f64>,white_imag:Vec<f64>,filter_len:usize,
    n:usize, fs:f64, bandwidth:f64, duration:f64, wavelength:f64,
    atmosphere:Option<(Vec<f64>,Vec<f64>)>,
    fft:Option<WhiteningFft>,
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
            whiten_real:r("whitener_real")?,whiten_imag:r("whitener_imag")?,white_real:Vec::new(),white_imag:Vec::new(),filter_len:h.dataset("whitener_real")?.shape()[1],
            atmosphere:if h.link_exists("height_m") {Some((r("height_m")?,r("density_kg_m3")?))} else {None},fft:None,time,real,n};
        if let Some((height,rho))=&v.atmosphere {
            anyhow::ensure!(height.len()>=2 && height.len()==rho.len() && height.windows(2).all(|w|w[1]>w[0]) && rho.iter().all(|r|r.is_finite() && *r>0.),"Invalid atmosphere");
        }
        anyhow::ensure!(v.center.len()==6 && v.frames.len()==42*v.time.len() &&
            v.rx.len()==3*v.time.len() && v.tx.len()==3 && v.sample_time.len()==v.real.len() &&
            v.imag.len()==v.real.len() && v.noise.len()==v.time.len() &&
            v.filter_len>0 && v.filter_len<v.n && v.whiten_real.len()==v.filter_len*v.time.len() && v.whiten_imag.len()==v.whiten_real.len(),"Invalid prepared arrays");
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
        v.prepare_fft();
        Ok(v)
    }
    fn prepare_fft(&mut self) {
        let nfft=(self.n+self.filter_len-1).next_power_of_two();
        let mut planner=FftPlanner::new();let forward=planner.plan_fft_forward(nfft);let inverse=planner.plan_fft_inverse(nfft);
        let mut filters=Vec::new();
        for i in 0..self.time.len() {
            let mut f=vec![Complex::new(0.,0.);nfft];
            for j in 0..self.filter_len {f[j]=Complex::new(self.whiten_real[i*self.filter_len+j],self.whiten_imag[i*self.filter_len+j]);}
            forward.process(&mut f);filters.push(f);
        }
        self.fft=Some(WhiteningFft {forward,inverse,filters});
    }
    fn frame(&self,i:usize,x:Vector3<f64>,v:Vector3<f64>)->(Vector3<f64>,Vector3<f64>) {
        let f=&self.frames[i*42..(i+1)*42]; let m=|s|Matrix3::from_row_slice(&f[s..s+9]);
        (Vector3::from_column_slice(&f[..3])+m(6)*x+m(24)*v,
         Vector3::from_column_slice(&f[3..6])+m(15)*x+m(33)*v)
    }
    pub fn residual(&self,q:&[f64],out:&mut[f64],log_norm:&mut f64)->bool {
        self.evaluate(q,out,log_norm,false)
    }
    fn dimension(&self)->usize {if self.atmosphere.is_some(){7}else{8}}
    fn observables(&self,q:&[f64])->Option<Vec<[f64;2]>> {
        if q.len()!=self.dimension() || q.iter().any(|x|!x.is_finite()) || q[..3].iter().any(|x|x.abs()>20.) || q[3..6].iter().any(|x|x.abs()>40.) {return None;}
        if self.atmosphere.is_some() && !(-7. ..=-2.).contains(&q[6]) {return None;}
        let x0=Vector3::from_fn(|j,_|self.center[j]+1000.*q[j]);
        let v0=Vector3::from_fn(|j,_|self.center[3+j]+1000.*q[3+j]);
        let speed=v0.norm();if !(1000. ..150000.).contains(&speed){return None;}
        let u=v0/speed;let tx=Vector3::from_column_slice(&self.tx);
        let states=self.states(q)?;let mut result=Vec::with_capacity(states.len());
        for (i,state) in states.iter().enumerate() {
            if !state[1].is_finite() || state[1]<=100. {return None;}
            let nominal=x0+u*state[0];let vel=u*state[1];let mut x=nominal;
            for _ in 0..2 {let (p,_)=self.frame(i,x,vel);x=nominal-vel*((p-tx).norm()/C);}
            let (p,w)=self.frame(i,x,vel);let rx=Vector3::from_column_slice(&self.rx[3*i..3*i+3]);
            let a=p-tx;let b=p-rx;
            result.push([a.norm()+b.norm(),-(a.normalize()+b.normalize()).dot(&w)/self.wavelength]);
        }
        Some(result)
    }
    fn states(&self,q:&[f64])->Option<Vec<[f64;4]>> {
        let x0=Vector3::from_fn(|j,_|self.center[j]+1000.*q[j]);
        let v0=Vector3::from_fn(|j,_|self.center[3+j]+1000.*q[3+j]);
        let speed=v0.norm();let u=v0/speed;
        if let Some((height,rho))=&self.atmosphere {
            let mut order:Vec<usize>=(0..self.time.len()).collect();
            order.sort_by(|&a,&b|self.time[a].total_cmp(&self.time[b]));
            let times:Vec<f64>=order.iter().map(|&i|self.time[i]).collect();
            let first=order[0];
            let density=|s:f64| {
                let (p,_)=self.frame(first,x0+u*s,Vector3::zeros());
                let z=crate::event::geodetic_height(p);
                if z<height[0] || z>*height.last().unwrap(){return f64::NAN;}
                let j=height.partition_point(|h|*h<z).clamp(1,height.len()-1);
                let w=(z-height[j-1])/(height[j]-height[j-1]);
                rho[j-1]*(1.-w)+rho[j]*w
            };
            let sorted=crate::shrinking_trajectory(&times,speed,10_f64.powf(q[6]),3000.,1e-8,density).ok()?;
            let mut states=vec![[0.;4];order.len()];
            for (j,&i) in order.iter().enumerate(){states[i]=sorted[j];}
            Some(states)
        } else {
            let a=10_f64.powf(q[6]);let b=10_f64.powf(q[7]);
            Some(self.time.iter().map(|&t|[speed*t-a*(b*t).exp_m1()/b,speed-a*(b*t).exp(),0.,0.]).collect())
        }
    }
    fn evaluate(&self,q:&[f64],out:&mut[f64],log_norm:&mut f64,prediction:bool)->bool {
        *log_norm=0.;
        if q.len()!=self.dimension() || q.iter().any(|x|!x.is_finite()) || q[..3].iter().any(|x|x.abs()>20.) ||
            q[3..6].iter().any(|x|x.abs()>40.) {return false;}
        if self.atmosphere.is_some() {if !(-7. ..=-2.).contains(&q[6]){return false;}}
        else if !(-5. ..=5.).contains(&q[6]) || !(-4. ..=3.).contains(&q[7]){return false;}
        let x0=Vector3::from_fn(|j,_|self.center[j]+1000.*q[j]);
        let v0=Vector3::from_fn(|j,_|self.center[3+j]+1000.*q[3+j]);
        let speed=v0.norm(); if !(1000. ..150000.).contains(&speed) {return false;}
        let u=v0/speed;
        let Some(states)=self.states(q) else{return false;};
        let tx=Vector3::from_column_slice(&self.tx);
        let mut model=vec![(0.,0.);self.n];
        let mut white_model=vec![(0.,0.);self.n];
        let mut spectrum=vec![Complex::new(0.,0.);self.fft.as_ref().map_or(0,|f|f.forward.len())];
        for (i,state) in states.iter().enumerate() {
            let v=state[1];
            if !v.is_finite() || v<=100. {return false;}
            let s=state[0];
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
            // Use only valid FIR rows: every input sample is inside the recorded
            // window. Apply the same linear transform to data and model, including
            // signal in the first filter_len-1 samples; those rows are not residuals.
            if let Some(fft)=&self.fft {
                spectrum.fill(Complex::new(0.,0.));
                for (j,m) in model.iter().enumerate(){spectrum[j]=Complex::new(m.0,m.1);}
                fft.forward.process(&mut spectrum);
                for (a,b) in spectrum.iter_mut().zip(&fft.filters[i]){*a*=b;}
                fft.inverse.process(&mut spectrum);
                let scale=1./spectrum.len() as f64;
                for j in self.filter_len-1..self.n {white_model[j]=(spectrum[j].re*scale,spectrum[j].im*scale);}
            }
            for j in self.filter_len-1..self.n {
                let (mut wr,mut wi)=white_model[j];
                if self.fft.is_none() {wr=0.;wi=0.;for lag in 0..self.filter_len {
                    let m=model[j-lag];let f=i*self.filter_len+lag;
                    wr+=self.whiten_real[f]*m.0-self.whiten_imag[f]*m.1;
                    wi+=self.whiten_real[f]*m.1+self.whiten_imag[f]*m.0;
                }}
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
#[no_mangle] pub unsafe extern "C" fn voltage_logp(v:*const Voltage,q:*const f64)->f64 {
    let v=&*v;let q=std::slice::from_raw_parts(q,v.dimension());let mut r=vec![0.;2*v.real.len()];let mut norm=0.;
    if v.residual(q,&mut r,&mut norm){-0.5*r.iter().map(|x|x*x).sum::<f64>()-norm}else{f64::NEG_INFINITY}
}
#[no_mangle] pub unsafe extern "C" fn voltage_states(v:*const Voltage,q:*const f64,out:*mut f64)->i32 {
    let v=&*v;
    if let Some(states)=v.states(std::slice::from_raw_parts(q,v.dimension())) {
        let out=std::slice::from_raw_parts_mut(out,4*v.time.len());
        for (i,s) in states.iter().enumerate(){out[4*i..4*i+4].copy_from_slice(s);}
        1
    }else{0}
}
#[no_mangle] pub unsafe extern "C" fn voltage_residual(v:*const Voltage,q:*const f64,out:*mut f64)->i32 {
    let v=&*v;
    v.residual(std::slice::from_raw_parts(q,v.dimension()),std::slice::from_raw_parts_mut(out,2*v.real.len()),&mut 0.) as i32
}
#[no_mangle] pub unsafe extern "C" fn voltage_free(v:*mut Voltage) {if !v.is_null(){drop(Box::from_raw(v));}}
#[no_mangle] pub unsafe extern "C" fn voltage_prediction(v:*const Voltage,q:*const f64,out:*mut f64)->i32 {
    let v=&*v;
    v.evaluate(std::slice::from_raw_parts(q,v.dimension()),std::slice::from_raw_parts_mut(out,2*v.real.len()),&mut 0.,true) as i32
}
#[no_mangle] pub unsafe extern "C" fn voltage_sample(v:*const Voltage,q:*const f64,widths:*const f64,
    warmup:usize,draws:usize,seed:u64,path:*const c_char)->i32 {
    let v=&*v;
    let logp=|q:&[f64]|{let mut r=vec![0.;2*v.real.len()];let mut log_norm=0.;
        // Integrate each complex amplitude with a flat Re/Im prior, rather than
        // treating its profiled estimate as known during posterior sampling.
        if v.residual(q,&mut r,&mut log_norm){-0.5*r.iter().map(|x|x*x).sum::<f64>()-log_norm}else{f64::NEG_INFINITY}};
    let result=(||->anyhow::Result<()> {
        let chain=metropolis(logp,std::slice::from_raw_parts(q,v.dimension()),std::slice::from_raw_parts(widths,v.dimension()),warmup,draws,seed)?;
        let h=hdf5::File::create(CStr::from_ptr(path).to_str()?)?;
        let flat:Vec<f64>=chain.samples.iter().flatten().copied().collect();
        h.new_dataset::<f64>().shape((draws,v.dimension())).create("samples")?.write_raw(&flat)?;
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
        let mut model=Voltage {center:vec![0.,0.,100000.,0.,0.,-10000.],time:vec![0.],frames:frame,
            rx:vec![0.;3],tx:vec![0.;3],sample_time,white_real:real.clone(),white_imag:imag.clone(),
            whiten_real:vec![1.],whiten_imag:vec![0.],filter_len:1,real,imag,noise:vec![1.],n,fs,bandwidth,duration,wavelength,atmosphere:None,fft:None};
        let mut q=vec![0.,0.,0.,0.,0.,0.,2.,0.];let mut residual=vec![0.;2*n];let mut lognorm=0.;
        assert!(model.residual(&q,&mut residual,&mut lognorm));
        assert!(residual.iter().map(|r|r*r).sum::<f64>()<1e-12);
        q[2]=0.001;
        assert!(model.residual(&q,&mut residual,&mut lognorm));
        assert!(residual.iter().map(|r|r*r).sum::<f64>()>0.001);
        // Match the direct complex FIR for a realistic 65-tap filter.
        model.filter_len=65;
        model.whiten_real=(0..65).map(|i|if i==0 {1.} else {0.01*(i as f64).cos()}).collect();
        model.whiten_imag=(0..65).map(|i|0.005*(i as f64).sin()).collect();
        assert!(model.residual(&q,&mut residual,&mut lognorm));
        let reference=residual.clone();let reference_norm=lognorm;
        model.prepare_fft();
        assert!(model.residual(&q,&mut residual,&mut lognorm));
        assert!((reference_norm-lognorm).abs()<1e-10);
        assert!(residual.iter().zip(&reference).all(|(a,b)|(a-b).abs()<1e-10));
        let beat=fd+bandwidth/duration*path/C;
        let reduced=reduced::Pulse::build(&model,0,beat);
        for shift in [-20000.,-5000.,-200.,-0.001,0.,0.001,200.,5000.,20000.] {
            for db in [-30000.,-5000.,-100.,0.,100.,5000.,30000.] {
                let exact=model.pulse_logp(0,path+shift,beat+db);
                let fast=reduced.logp(path+shift,beat+db);
                assert!((exact-fast).abs()<1e-6,"path {shift} beat {db}: {exact} vs {fast}");
            }
        }
        // Cropped echoes may start before the first valid residual row. The
        // transformed data still have a valid model; do not reject these pulses.
        for sample in &mut model.sample_time {*sample+=100./fs;}
        model.fft=None;
        assert!(model.residual(&q,&mut residual,&mut lognorm));
        let reference=residual.clone();model.prepare_fft();
        assert!(model.residual(&q,&mut residual,&mut lognorm));
        assert!(residual.iter().zip(&reference).all(|(a,b)|(a-b).abs()<1e-10));
    }
}
