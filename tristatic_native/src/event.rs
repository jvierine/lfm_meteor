//! Prepared inputs contain fixed MSIS/frame transforms, never Python callbacks.
use anyhow::{bail, Result};
use nalgebra::{Matrix3, Vector3};
use crate::{chirp_observables, paired_t_loglike, shrinking_trajectory, C};

pub struct Event {
    pub time:Vec<f64>, pub center:Vec<f64>, pub initial:Vec<f64>, pub widths:Vec<f64>,
    path:Vec<f64>, beat:Vec<f64>, anchor:Vec<f64>, scale:Vec<f64>, keep:Vec<f64>,
    frames:Vec<f64>, sites:Vec<f64>, height:Vec<f64>, density:Vec<f64>,
    wavelength:f64, chirp:f64, nu:f64,
}
fn read(h:&hdf5::File,key:&str)->Result<Vec<f64>> {Ok(h.dataset(key)?.read_raw::<f64>()?)}
impl Event {
    pub fn load(path:&str)->Result<Self> {
        let h=hdf5::File::open(path)?;
        let e=Self {time:read(&h,"time_s")?,center:read(&h,"center")?,
            initial:read(&h,"initial")?,widths:read(&h,"widths")?,
            path:read(&h,"path_m")?,beat:read(&h,"beat_hz")?,anchor:read(&h,"anchor_m")?,
            scale:read(&h,"scale_matrix")?,keep:read(&h,"keep")?,frames:read(&h,"frames")?,
            sites:read(&h,"sites")?,height:read(&h,"height_m")?,density:read(&h,"density_kg_m3")?,
            wavelength:h.dataset("wavelength_m")?.read_scalar()?,
            chirp:h.dataset("chirp_hz_s")?.read_scalar()?,nu:h.dataset("nu")?.read_scalar()?};
        let n=e.time.len();
        if n==0 || e.time[0]!=0. || e.center.len()!=6 || e.initial.len()!=16 || e.widths.len()!=16 ||
            e.path.len()!=3*n || e.beat.len()!=3*n || e.anchor.len()!=3*n || e.keep.len()!=3*n ||
            e.scale.len()!=9*n || e.frames.len()!=42*n || e.sites.len()!=12 || e.height.len()<2 ||
            e.height.len()!=e.density.len() || e.height.windows(2).any(|w|w[1]<=w[0]) ||
            e.nu<=2. || e.chirp<=0. || e.wavelength<=0. {bail!("Invalid prepared event schema");}
        Ok(e)
    }
    fn frame(&self,i:usize,x:Vector3<f64>,v:Vector3<f64>)->(Vector3<f64>,Vector3<f64>) {
        let f=&self.frames[i*42..(i+1)*42];
        let m=|start|Matrix3::from_row_slice(&f[start..start+9]);
        (Vector3::from_column_slice(&f[..3])+m(6)*x+m(24)*v,
         Vector3::from_column_slice(&f[3..6])+m(15)*x+m(33)*v)
    }
    fn rho(&self,x:Vector3<f64>)->f64 {
        let (p,_)=self.frame(0,x,Vector3::zeros());
        let h=geodetic_height(p);
        let j=self.height.partition_point(|z|*z<h);
        if j==0 {return self.density[0];}
        if j==self.height.len() {return *self.density.last().unwrap();}
        let w=(h-self.height[j-1])/(self.height[j]-self.height[j-1]);
        self.density[j-1]*(1.-w)+self.density[j]*w
    }
    pub fn prediction(&self,q:&[f64])->Result<Vec<[f64;2]>> {
        let x=Vector3::from_fn(|i,_|self.center[i]+1000.*q[i]);
        let v=Vector3::from_fn(|i,_|self.center[3+i]+1000.*q[3+i]);
        let speed=v.norm(); let u=v/speed;
        let states=shrinking_trajectory(&self.time,speed,10_f64.powf(q[6]),3000.,1e-8,
            |s|self.rho(x+u*s))?;
        let tx=Vector3::from_column_slice(&self.sites[..3]);
        let mut result=Vec::with_capacity(3*self.time.len());
        for (i,s) in states.iter().enumerate() {
            let nominal=x+u*s[0]; let velocity=u*s[1]; let mut retarded=nominal;
            // Preserve existing catalogue timing pending independent timestamp audit.
            for _ in 0..2 {
                let (p,_)=self.frame(i,retarded,velocity);
                retarded=nominal-velocity*((p-tx).norm()/C);
            }
            let (p,w)=self.frame(i,retarded,velocity);
            for station in 0..3 {
                let rx=Vector3::from_column_slice(&self.sites[3+station*3..6+station*3]);
                let a=p-tx; let b=p-rx; let path=a.norm()+b.norm();
                let rate=(a/a.norm()+b/b.norm()).dot(&w);
                let (delay,beat)=chirp_observables(path,rate,self.wavelength,self.chirp,self.anchor[i*3+station]);
                result.push([delay,beat+1000.*q[7+station]]);
            }
        }
        Ok(result)
    }
    pub fn logp(&self,q:&[f64])->f64 {
        if q.len()!=16 || q.iter().any(|v|!v.is_finite()) || q[..3].iter().any(|v|v.abs()>20.) ||
            q[3..6].iter().enumerate().any(|(i,v)|(self.center[i+3]+1000.*v).abs()>80000.) ||
            !(-7. ..=-2.).contains(&q[6]) || q[7..10].iter().any(|v|v.abs()>500.) ||
            q[10..].iter().any(|v|v.abs()>4.) {return f64::NEG_INFINITY;}
        let Ok(prediction)=self.prediction(q) else {return f64::NEG_INFINITY;};
        let mut lp=q[10..].iter().map(|x|-0.5*(x/3_f64.ln()).powi(2)).sum::<f64>();
        for (j,p) in prediction.iter().enumerate() {
            if self.keep[j]==0. {continue;}
            let station=j%3; let a=q[10+2*station].exp(); let b=q[11+2*station].exp();
            let s=&self.scale[3*j..3*j+3];
            lp+=paired_t_loglike([p[0]-self.path[j],p[1]-self.beat[j]],
                [s[0]*a*a,s[1]*a*b,s[2]*b*b],self.nu);
        }
        if lp.is_finite() {lp} else {f64::NEG_INFINITY}
    }
}

pub fn geodetic_height(p:Vector3<f64>)->f64 {
    let a=6378137.; let e2=6.6943799901413165e-3; let r=p[0].hypot(p[1]);
    if r<1e-6 {return p[2].abs()-a*(1_f64-e2).sqrt();}
    let mut latitude=p[2].atan2(r*(1.-e2));
    for _ in 0..8 {
        let n=a/(1.-e2*latitude.sin().powi(2)).sqrt();
        latitude=(p[2]+e2*n*latitude.sin()).atan2(r);
    }
    let n=a/(1.-e2*latitude.sin().powi(2)).sqrt();
    r/latitude.cos()-n
}
