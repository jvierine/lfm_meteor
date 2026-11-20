//! Sufficient statistics for the sampled rectangular chirp and valid FIR rows.
//! Path enters only through its envelope; free complex amplitude absorbs phase.
use super::*;
const ORDER:usize=48;
type Z=Complex<f64>;

pub(super) struct Pulse {
    n:usize, f:usize, fs:f64, t0:f64, duration:f64, beat:f64, half_time:f64,
    cross:Vec<Z>, coefficients:Vec<Z>, gram:Vec<Z>, noise:f64,
}
impl Pulse {
    pub(super) fn build(v:&Voltage,i:usize,beat:f64)->Self {
        let n=v.n;let f=v.filter_len;let zero=Z::new(0.,0.);
        let t0=v.sample_time[i*n];let tc=t0+(n-1) as f64/(2.*v.fs);
        let half_time=(n-1) as f64/(2.*v.fs);
        let p:Vec<Z>=(0..n).map(|j| {
            let t=t0+j as f64/v.fs;
            let phase=std::f64::consts::TAU*(-v.bandwidth/(2.*v.duration)*t*t+(v.bandwidth/2.+beat)*t);
            Z::from_polar(1.,phase)
        }).collect();
        let h:Vec<Z>=(0..f).map(|l|Z::new(v.whiten_real[i*f+l],v.whiten_imag[i*f+l])).collect();
        let mut cross=vec![zero;n];
        for r in f-1..n {
            let d=Z::new(v.white_real[i*n+r],v.white_imag[i*n+r]);
            for l in 0..f {cross[r-l]+=h[l].conj()*d;}
        }
        for j in 0..n {cross[j]*=p[j].conj();}
        let mut coefficients=vec![zero;ORDER*(n+1)];
        for j in 0..n {
            let s=(t0+j as f64/v.fs-tc)/half_time;
            let mut a=cross[j];
            for k in 0..ORDER {
                coefficients[k*(n+1)+j+1]=coefficients[k*(n+1)+j]+a;
                a*=Z::new(0.,-s)/(k+1) as f64;
            }
        }
        let mut gram=vec![zero;f*(n+1)];
        for lag in 0..f {
            let mut hp=vec![zero;f+1];
            for l in lag..f {hp[l+1]=hp[l]+h[l].conj()*h[l-lag];}
            for j in 0..n-lag {
                let lo=lag.max((f-1).saturating_sub(j));let hi=(f-1).min(n-1-j);
                let q=if lo<=hi {hp[hi+1]-hp[lo]}else{zero};
                gram[lag*(n+1)+j+1]=gram[lag*(n+1)+j]+p[j].conj()*q*p[j+lag];
            }
        }
        Self{n,f,fs:v.fs,t0,duration:v.duration,beat,half_time,cross,coefficients,gram,noise:v.noise[i]}
    }
    fn envelope(&self,j:isize,a:f64)->f64 {
        (j as f64-a+0.5).clamp(0.,1.)*(a+self.duration*self.fs-j as f64+0.5).clamp(0.,1.)
    }
    fn weighted(&self,prefix:&[Z],a:f64,lag:usize,square:bool)->Z {
        let first=(a+0.5).ceil() as isize;let last=(a+self.duration*self.fs-0.5).floor() as isize;
        let lo=first.max(0);let hi=(last-lag as isize).min((self.n-lag) as isize-1);
        let mut sum=if lo<=hi {prefix[hi as usize+1]-prefix[lo as usize]}else{Z::new(0.,0.)};
        let left=(a+0.5).floor() as isize;let right=(a+self.duration*self.fs-0.5).ceil() as isize;
        let candidates=[left,right,left-lag as isize,right-lag as isize];
        for (k,&j) in candidates.iter().enumerate() {
            if j<0 || j>=(self.n-lag) as isize || (lo<=j && j<=hi) || candidates[..k].contains(&j){continue;}
            let e=self.envelope(j,a)*if square {self.envelope(j+lag as isize,a)}else{1.};
            sum+=(prefix[j as usize+1]-prefix[j as usize])*e;
        }
        sum
    }
    pub(super) fn logp(&self,path:f64,beat:f64)->f64 {
        let a=(path/C-self.t0)*self.fs;let delta=std::f64::consts::TAU*(beat-self.beat)*self.half_time;
        let cross=if delta.abs()<=6. {
            let mut value=Z::new(0.,0.);
            for k in (0..ORDER).rev() {value=value*delta+self.weighted(&self.coefficients[k*(self.n+1)..(k+1)*(self.n+1)],a,0,false);}
            value
        }else{
            // Exact fallback, not a posterior truncation at the expansion radius.
            let mut value=Z::new(0.,0.);
            for j in 0..self.n {let e=self.envelope(j as isize,a);if e>0. {
                let s=2.*j as f64/(self.n-1) as f64-1.;
                value+=self.cross[j]*Z::from_polar(e,-delta*s);
            }}value
        };
        let mut norm=self.weighted(&self.gram[..self.n+1],a,0,true).re;
        let step=Z::from_polar(1.,std::f64::consts::TAU*(beat-self.beat)/self.fs);let mut phase=step;
        for lag in 1..self.f {
            norm+=2.*(phase*self.weighted(&self.gram[lag*(self.n+1)..(lag+1)*(self.n+1)],a,lag,true)).re;
            phase*=step;
        }
        if norm<10. || !norm.is_finite(){return f64::NEG_INFINITY;}
        cross.norm_sqr()/(self.noise*norm)-norm.ln()
    }
}
