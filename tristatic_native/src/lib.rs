//! Native trajectory, explicit joint likelihood, and Metropolis-Hastings kernels.
pub mod event;
pub mod voltage;
use anyhow::{bail, Result};
use nalgebra::{DMatrix, DVector, Vector2};
use ode_solvers::{continuous_output_model::ContinuousOutputModel, Dopri5, System};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
pub const C: f64 = 299_792_458.;

pub fn chirp_observables(path:f64, rate:f64, wavelength:f64, chirp:f64, anchor:f64)->(f64,f64) {
    let doppler = -rate/wavelength;
    (path+C*doppler/chirp, doppler+chirp/C*(path-anchor))
}

/// Scale matrix is not covariance: covariance is nu/(nu-2) times scale, for nu>2.
/// Omitted normalization depends only on nu, which must remain fixed.
pub fn paired_t_loglike(r:[f64;2], scale:[f64;3], nu:f64)->f64 {
    let [a,b,d]=scale; let det=a*d-b*b; let [x,y]=r;
    if a<=0. || d<=0. || det<=0. || nu<=0. {return f64::NEG_INFINITY;}
    let q=(d*x*x-2.*b*x*y+a*y*y)/det;
    -0.5*det.ln()-(nu+2.)/2.*(q/nu).ln_1p()
}

pub struct Chain {
    pub samples:Vec<Vec<f64>>, pub log_probability:Vec<f64>,
    pub accepted:usize, pub proposal_cholesky:DMatrix<f64>,
}

/// Adapt empirical proposal covariance during warmup only; retained MH is homogeneous.
pub fn metropolis<F:Fn(&[f64])->f64>(logp:F, initial:&[f64], widths:&[f64],
                                    warmup:usize, draws:usize, seed:u64)->Result<Chain> {
    let n=initial.len();
    if n==0 || widths.len()!=n || widths.iter().any(|s| !s.is_finite() || *s<=0.) {
        bail!("Invalid proposal scales");
    }
    let mut rng=StdRng::seed_from_u64(seed);
    let mut x=DVector::from_column_slice(initial); let mut lp=logp(x.as_slice());
    if !lp.is_finite() {bail!("Nonfinite initial posterior");}
    let mut chol=DMatrix::from_diagonal(&DVector::from_column_slice(widths));
    let mut mean=DVector::zeros(n); let mut scatter=DMatrix::zeros(n,n);
    let mut accepted=0; let mut samples=Vec::with_capacity(draws); let mut logs=Vec::with_capacity(draws);
    for step in 0..warmup+draws {
        let z=DVector::from_fn(n, |_,_| rng.sample::<f64,_>(StandardNormal));
        let proposal=&x+&chol*z; let candidate=logp(proposal.as_slice());
        if candidate.is_finite() && rng.gen::<f64>().ln()<candidate-lp {
            x=proposal; lp=candidate; if step>=warmup {accepted+=1;}
        }
        if step<warmup {
            let count=(step+1) as f64; let delta=&x-&mean;
            mean+=&delta/count; scatter+=&delta*(&x-&mean).transpose();
            if step+1>=100.max(10*n) && (step+1)%50==0 {
                let mut covariance=&scatter/(count-1.)*(2.38_f64.powi(2)/n as f64);
                for j in 0..n {covariance[(j,j)]+=0.01*widths[j]*widths[j];}
                if let Some(factor)=covariance.cholesky() {chol=factor.l();}
            }
        } else {samples.push(x.as_slice().to_vec()); logs.push(lp);}
    }
    Ok(Chain {samples,log_probability:logs,accepted,proposal_cholesky:chol})
}

struct Shrinking<'a,F> {density:&'a F, speed0:f64, radius0:f64, material:f64, sigma:f64}
impl<F:Fn(f64)->f64> System<f64,Vector2<f64>> for Shrinking<'_,F> {
    fn system(&self,_:f64,y:&Vector2<f64>,dy:&mut Vector2<f64>) {
        let r=self.radius0*(self.sigma*(y[1]*y[1]-self.speed0*self.speed0)/6.).exp();
        dy[0]=y[1]; dy[1]=-3.*(self.density)(y[0])*y[1]*y[1]/(4.*self.material*r);
    }
}

/// Fixed direction, Gamma=1; density(s) must use the trial position's geodetic height.
/// Return distance, speed, radius, mass. Radius follows the exact mass-speed invariant.
pub fn shrinking_trajectory<F:Fn(f64)->f64>(times:&[f64],speed0:f64,radius0:f64,
        material:f64,sigma:f64,density:F)->Result<Vec<[f64;4]>> {
    if times.is_empty() || times.iter().any(|t| !t.is_finite() || *t<0.) ||
        times.windows(2).any(|w|w[1]<w[0]) || speed0<=0. || radius0<=0. || material<=0. || sigma<0. {
        bail!("Invalid trajectory inputs");
    }
    let end=*times.last().unwrap(); let mut dense=ContinuousOutputModel::default();
    if end>0. {
        let model=Shrinking {density:&density,speed0,radius0,material,sigma};
        let mut solver=Dopri5::new(model,0.,end,0.0005,Vector2::new(0.,speed0),1e-10,1e-8);
        solver.integrate_with_continuous_output_model(&mut dense)
            .map_err(|e|anyhow::anyhow!("ODE failed: {e:?}"))?;
    }
    times.iter().map(|&t| {
        let y=if t==0. {Vector2::new(0.,speed0)} else {
            dense.evaluate(t).ok_or_else(||anyhow::anyhow!("Missing dense output"))?
        };
        let radius=radius0*(sigma*(y[1]*y[1]-speed0*speed0)/6.).exp();
        let mass=4.*std::f64::consts::PI/3.*material*radius.powi(3);
        if !y.iter().all(|v|v.is_finite()) || y[1]<=0. || radius<=0. || !mass.is_finite() {
            bail!("Nonphysical trajectory");
        }
        Ok([y[0],y[1],radius,mass])
    }).collect()
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn beat_delay_identity() {
        let (a,b)=chirp_observables(200000.,50000.,0.7,2e10,199000.);
        assert!((b-2e10/C*(a-199000.)).abs()<1e-8);
    }
    #[test] fn correlated_normal() {
        let c=metropolis(|x|-(x[0]*x[0]-1.6*x[0]*x[1]+x[1]*x[1])/(2.*0.36),
            &[0.,0.],&[0.5,0.5],3000,40000,42).unwrap();
        let n=c.samples.len() as f64;
        let m:Vec<f64>=(0..2).map(|i|c.samples.iter().map(|x|x[i]).sum::<f64>()/n).collect();
        assert!(m.iter().all(|x|x.abs()<0.07));
        let cov=c.samples.iter().map(|x|(x[0]-m[0])*(x[1]-m[1])).sum::<f64>()/n;
        assert!((cov-0.8).abs()<0.08);
    }
    #[test] fn bounded_prior() {
        let c=metropolis(|x|if (0. ..=1.).contains(&x[0]) {0.} else {f64::NEG_INFINITY},
            &[0.5],&[0.3],1000,10000,3).unwrap();
        assert!(c.samples.iter().all(|x|(0. ..=1.).contains(&x[0])));
        assert!((c.samples.iter().map(|x|x[0]).sum::<f64>()/10000.-0.5).abs()<0.03);
    }
    #[test] fn no_ablation_analytic_solution() {
        let rho=1e-7; let r=50e-6; let v=40000.; let material=3000.;
        let k=3.*rho/(4.*material*r);
        let result=shrinking_trajectory(&[0.,0.1,0.5],v,r,material,0.,|_|rho).unwrap();
        for (&t,y) in [0.,0.1,0.5].iter().zip(result) {
            assert!((y[1]-v/(1.+k*v*t)).abs()<1e-4);
            assert!((y[0]-(k*v*t).ln_1p()/k).abs()<1e-4);
            assert_eq!(y[2],r);
        }
    }
    #[test] fn mass_speed_invariant() {
        let v=40000.; let r=50e-6; let sigma=1e-8;
        let result=shrinking_trajectory(&[0.,0.1,0.2],v,r,3000.,sigma,|_|1e-7).unwrap();
        for y in &result {
            assert!(((y[3]/result[0][3]).ln()-sigma/2.*(y[1]*y[1]-v*v)).abs()<1e-12);
        }
        assert!(result.windows(2).all(|w|w[1][1]<w[0][1] && w[1][2]<w[0][2]));
    }
}
