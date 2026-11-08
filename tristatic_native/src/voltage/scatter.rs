//! Independent per-pulse physical delay/Doppler scatter, not receiver-noise inflation.
//! Exact sampled-voltage likelihood; local Gaussian fits are MH proposals only.
use super::*;
use super::compressed::{RadiusTransport, tune_scale};
use nalgebra::{DMatrix, DVector, Matrix2, Vector2};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
use std::time::Instant;

type V = Vector2<f64>;
type M = Matrix2<f64>;

struct Datum {pulse:reduced::Pulse, origin:V, maximum:V, precision:M, offset:f64, station:usize}
struct Scatter {data:Vec<Datum>, path_scale:f64, beat_scale:f64}

fn quadratic(x:V, mean:V, precision:M)->f64 {
    let d=x-mean; -0.5*d.dot(&(precision*d))
}

fn prior_precision(log_s:[f64;2])->M {
    let a=(-2.*log_s[0]).exp(); let b=(-2.*log_s[1]).exp();
    // In chirp units, residual Doppler is residual beat minus residual path.
    M::new(a+b,-b,-b,b)
}

fn nuisance_logprior(log_s:&[f64])->f64 {
    // Independent half-Student-t(3), one range/frequency resolution scale.
    log_s.iter().map(|&x|x-2.*(1.+(2.*x).exp()/3.).ln()).sum()
}

fn gaussian_draw(mean:V, covariance:M, rng:&mut StdRng)->V {
    let l=covariance.cholesky().unwrap().l();
    mean+l*V::new(rng.sample(StandardNormal),rng.sample(StandardNormal))
}

fn latent_step<F:Fn(V)->f64>(x:V, previous_ll:f64, mean:V, prior:M,
    maximum:V, measurement:M, ll:F, rng:&mut StdRng)->(V,f64,bool) {
    let guided=rng.gen::<f64>()>=0.05;
    let (proposal_mean,covariance)=if guided {
        let covariance=(prior+measurement).try_inverse().unwrap();
        (covariance*(prior*mean+measurement*maximum),covariance)
    } else {(mean,prior.try_inverse().unwrap())};
    let proposal=gaussian_draw(proposal_mean,covariance,rng);
    let next_ll=ll(proposal);
    let correction=if guided {
        quadratic(proposal,maximum,measurement)-quadratic(x,maximum,measurement)
    } else {0.};
    let accept=next_ll.is_finite() && rng.gen::<f64>().ln()<next_ll-previous_ll-correction;
    if accept {(proposal,next_ll,true)}else{(x,previous_ll,false)}
}

impl Scatter {
    fn build(v:&Voltage,q:&[f64],stations:&[u64])->anyhow::Result<Self> {
        let obs=v.observables(q).ok_or_else(||anyhow::anyhow!("Invalid physical seed"))?;
        anyhow::ensure!(stations.len()==obs.len() && stations.iter().all(|&s|s<3),"Invalid stations");
        let path_scale=C/v.bandwidth; let beat_scale=1./v.duration;
        let chirp=v.bandwidth/v.duration/C;
        let mut data=Vec::new();
        for (i,o) in obs.iter().enumerate() {
            let origin=V::new(o[0],o[1]+chirp*o[0]);
            let pulse=reduced::Pulse::build(v,i,origin[1]);
            let offset=pulse.logp(origin[0],origin[1]);
            let ll=|x:V|pulse.logp(origin[0]+path_scale*x[0],origin[1]+beat_scale*x[1])-offset;
            let mut simplex=[V::zeros(),V::new(0.05,0.),V::new(0.,0.02)];
            for _ in 0..500 {
                simplex.sort_by(|a,b|ll(*b).total_cmp(&ll(*a)));
                if (simplex[2]-simplex[0]).norm()<1e-6 {break;}
                let center=(simplex[0]+simplex[1])/2.;let reflect=2.*center-simplex[2];
                if ll(reflect)>ll(simplex[0]) {
                    let expand=center+2.*(reflect-center);
                    simplex[2]=if ll(expand)>ll(reflect){expand}else{reflect};
                } else if ll(reflect)>ll(simplex[1]) {simplex[2]=reflect;}
                else {let contract=(center+simplex[2])/2.;
                    if ll(contract)>ll(simplex[2]) {simplex[2]=contract;}
                    else {simplex[1]=(simplex[0]+simplex[1])/2.;simplex[2]=(simplex[0]+simplex[2])/2.;}}
            }
            simplex.sort_by(|a,b|ll(*b).total_cmp(&ll(*a)));
            let maximum=simplex[0];let step=0.001;let f=ll(maximum);
            let e=[V::new(step,0.),V::new(0.,step)];
            let a=-(ll(maximum+e[0])-2.*f+ll(maximum-e[0]))/(step*step);
            let d=-(ll(maximum+e[1])-2.*f+ll(maximum-e[1]))/(step*step);
            let b=-(ll(maximum+e[0]+e[1])-ll(maximum+e[0]-e[1])
                  -ll(maximum-e[0]+e[1])+ll(maximum-e[0]-e[1]))/(4.*step*step);
            anyhow::ensure!([a,b,d,f].iter().all(|x|x.is_finite()),"Invalid local proposal for pulse {i}");
            // A curvature floor stabilizes proposals only; it never alters ll.
            let eig=M::new(a,b,b,d).symmetric_eigen();
            let floor=eig.eigenvalues.max().max(1.)*1e-8;
            let precision=eig.eigenvectors*M::from_diagonal(&eig.eigenvalues.map(|x|x.max(floor)))*eig.eigenvectors.transpose();
            data.push(Datum{pulse,origin,maximum,precision,offset,station:stations[i] as usize});
        }
        Ok(Self{data,path_scale,beat_scale})
    }
    fn ll(&self,i:usize,x:V)->f64 {
        let d=&self.data[i];
        d.pulse.logp(d.origin[0]+self.path_scale*x[0],d.origin[1]+self.beat_scale*x[1])-d.offset
    }
    fn means(&self,v:&Voltage,q:&[f64])->Option<Vec<V>> {
        let chirp=v.bandwidth/v.duration/C;
        Some(v.observables(q)?.iter().zip(&self.data).map(|(o,d)|
            V::new((o[0]-d.origin[0])/self.path_scale,
                   (o[1]+chirp*o[0]-d.origin[1])/self.beat_scale)).collect())
    }
    fn residual_logp(&self,values:&[V],means:&[V],logs:&[f64],station:Option<usize>)->f64 {
        self.data.iter().enumerate().filter(|(_,d)|station.map_or(true,|s|d.station==s)).map(|(i,d)| {
            let s=d.station;let precision=prior_precision([logs[2*s],logs[2*s+1]]);
            quadratic(values[i],means[i],precision)-logs[2*s]-logs[2*s+1]
        }).sum()
    }
}

#[no_mangle]
pub unsafe extern "C" fn voltage_scatter_sample(v:*const Voltage,stations:*const u64,q:*const f64,
    covariance:*const f64,warmup:usize,draws:usize,seconds:f64,seed:u64,path:*const c_char)->i32 {
    let result=(||->anyhow::Result<()> {
        let v=&*v;let q=std::slice::from_raw_parts(q,7);let stations=std::slice::from_raw_parts(stations,v.time.len());
        let prepared=Instant::now();let model=Scatter::build(v,q,stations)?;
        let mut rng=StdRng::seed_from_u64(seed);let mut x=DVector::from_column_slice(q);
        let base=DMatrix::from_row_slice(7,7,std::slice::from_raw_parts(covariance,49));
        let transport=RadiusTransport::new(&x,&base);
        let mut physical=transport.physical(&x);let mut means=model.means(v,physical.as_slice()).unwrap();
        let mut values:Vec<V>=model.data.iter().map(|d|d.maximum).collect();
        let mut pulse_ll:Vec<f64>=(0..values.len()).map(|i|model.ll(i,values[i])).collect();
        let mut logs=[-3.;6];
        for s in 0..3 {for j in 0..2 {
            let sum:Vec<f64>=model.data.iter().enumerate().filter(|(_,d)|d.station==s).map(|(i,_)| {
                let r=values[i]-means[i];if j==0 {r[0]}else{r[1]-r[0]}
            }).collect();
            logs[2*s+j]=(sum.iter().map(|r|r*r).sum::<f64>()/sum.len() as f64).sqrt().clamp(0.001,2.).ln();
        }}
        let mut chol=(&base*100.).cholesky().ok_or_else(||anyhow::anyhow!("Invalid proposal covariance"))?.l()*(2.38/7_f64.sqrt());
        let mut running_mean=DVector::zeros(7);let mut scatter=DMatrix::zeros(7,7);
        let mut scale=0_f64;let mut hyper_scale=[0_f64;3];let mut accepted=[0u64;5];
        let mut samples=Vec::new();let mut nuisance=Vec::new();let mut probability=Vec::new();
        let mut latent_sum=vec![V::zeros();values.len()];let mut latent_count=0usize;
        let elapsed_prepare=prepared.elapsed().as_secs_f64();let started=Instant::now();
        for step in 0..warmup+draws {
            if started.elapsed().as_secs_f64()>=seconds {break;}
            let z=DVector::from_fn(7,|_,_|rng.sample(StandardNormal));
            let proposed=&x+(&chol*z)*scale.exp();let proposed_physical=transport.physical(&proposed);
            let current_prior=model.residual_logp(&values,&means,&logs,None);
            let proposed_means=model.means(v,proposed_physical.as_slice());
            let proposed_prior=proposed_means.as_ref().map_or(f64::NEG_INFINITY,|m|model.residual_logp(&values,m,&logs,None));
            let accept=proposed_prior.is_finite() && rng.gen::<f64>().ln()<proposed_prior-current_prior;
            if accept {x=proposed;physical=proposed_physical;means=proposed_means.unwrap();if step>=warmup {accepted[0]+=1;}}
            if step<warmup {
                scale=tune_scale(scale,step,accept);
                let count=(step+1) as f64;let delta=&x-&running_mean;running_mean+=&delta/count;
                scatter+=&delta*(&x-&running_mean).transpose();
                if step>=499 && step%100==99 {
                    let cov=(&scatter/(count-1.)+&base*0.01)*(2.38_f64.powi(2)/7.);
                    if let Some(c)=cov.cholesky(){chol=c.l();}
                }
            }
            for i in 0..values.len() {
                let d=&model.data[i];let s=d.station;
                let (next,next_ll,ok)=latent_step(values[i],pulse_ll[i],means[i],
                    prior_precision([logs[2*s],logs[2*s+1]]),d.maximum,d.precision,
                    |u|model.ll(i,u),&mut rng);
                values[i]=next;pulse_ll[i]=next_ll;if ok && step>=warmup {accepted[4]+=1;}
            }
            for s in 0..3 {
                let mut candidate=logs;
                for j in 0..2 {candidate[2*s+j]+=0.2*hyper_scale[s].exp()*rng.sample::<f64,_>(StandardNormal);}
                let old=model.residual_logp(&values,&means,&logs,Some(s))+nuisance_logprior(&logs[2*s..2*s+2]);
                let new=model.residual_logp(&values,&means,&candidate,Some(s))+nuisance_logprior(&candidate[2*s..2*s+2]);
                let ok=new.is_finite() && rng.gen::<f64>().ln()<new-old;
                if ok {logs=candidate;if step>=warmup {accepted[1+s]+=1;}}
                if step<warmup {hyper_scale[s]=tune_scale(hyper_scale[s],step,ok);}
            }
            if step>=warmup {
                samples.extend_from_slice(physical.as_slice());
                for s in 0..3 {nuisance.push(logs[2*s].exp()*model.path_scale);nuisance.push(logs[2*s+1].exp()*model.beat_scale);}
                probability.push(pulse_ll.iter().sum::<f64>()+model.residual_logp(&values,&means,&logs,None)+nuisance_logprior(&logs));
                if step%20==0 {for (sum,value) in latent_sum.iter_mut().zip(&values){*sum+=value;}latent_count+=1;}
            }
        }
        let h=hdf5::File::create(CStr::from_ptr(path).to_str()?)?;
        h.new_dataset::<f64>().shape((probability.len(),7)).create("samples")?.write_raw(&samples)?;
        h.new_dataset::<f64>().shape((probability.len(),6)).create("scatter_samples")?.write_raw(&nuisance)?;
        h.new_dataset_builder().with_data(&probability).create("log_probability")?;
        h.new_dataset_builder().with_data(&accepted).create("accepted")?;
        h.new_dataset::<f64>().create("elapsed_seconds")?.write_scalar(&started.elapsed().as_secs_f64())?;
        h.new_dataset::<f64>().create("preparation_seconds")?.write_scalar(&elapsed_prepare)?;
        h.new_dataset::<u64>().create("latent_count")?.write_scalar(&(latent_count as u64))?;
        let mut fitted=Vec::new();
        for (sum,d) in latent_sum.iter().zip(&model.data) {
            let u=sum/latent_count.max(1) as f64;
            fitted.extend_from_slice(&[d.origin[0]+model.path_scale*u[0],d.origin[1]+model.beat_scale*u[1]]);
        }
        h.new_dataset::<f64>().shape((values.len(),2)).create("mean_pulse_path_and_absolute_beat")?.write_raw(&fitted)?;
        h.flush()?;Ok(())
    })();match result {Ok(())=>1,Err(e)=>{eprintln!("{e:#}");0}}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chirp_coupling_and_normalization() {
        let logs=[0.7_f64.ln(),0.2_f64.ln()];let p=prior_precision(logs);
        let k=p.try_inverse().unwrap();
        assert!((k[(0,0)]-0.49).abs()<1e-12);
        assert!((k[(0,1)]-0.49).abs()<1e-12);
        assert!((k[(1,1)]-0.53).abs()<1e-12);
        assert!((k.determinant()-0.49*0.04).abs()<1e-12);
    }
    #[test]
    fn latent_mh_recovers_exact_correlated_gaussian() {
        let mean=V::new(0.2,-0.3);let maximum=V::new(0.8,0.6);
        let p=prior_precision([0.7_f64.ln(),0.4_f64.ln()]);
        let measurement=M::new(15.,-3.,-3.,12.);
        let covariance=(p+measurement).try_inverse().unwrap();
        let truth=covariance*(p*mean+measurement*maximum);
        let mut rng=StdRng::seed_from_u64(1846);let mut x=mean;
        let ll=|u|quadratic(u,maximum,measurement);let mut previous=ll(x);
        let mut sum=V::zeros();let mut squares=M::zeros();let n=40000;
        for _ in 0..n {
            let (y,l,_)=latent_step(x,previous,mean,p,maximum,measurement,ll,&mut rng);x=y;previous=l;
            sum+=x;squares+=x*x.transpose();
        }
        let estimated=sum/n as f64;let estimated_cov=squares/n as f64-estimated*estimated.transpose();
        assert!((estimated-truth).norm()<0.01,"{estimated:?} {truth:?}");
        assert!((estimated_cov-covariance).norm()<0.005);
    }
    #[test]
    fn approximate_proposal_does_not_change_non_gaussian_target() {
        // Importance moments provide an independent reference for a quartic likelihood.
        let p=M::identity();let mean=V::zeros();let proposal=M::identity()*3.;
        let ll=|u:V|-0.5*(u[0]-0.4).powi(2)-0.6*u[0].powi(4)-1.2*(u[1]+0.2).powi(2);
        let mut rng=StdRng::seed_from_u64(77);let mut norm=0.;let mut truth=V::zeros();
        for _ in 0..200000 {let u=gaussian_draw(mean,M::identity(),&mut rng);let w=ll(u).exp();norm+=w;truth+=u*w;}
        truth/=norm;let mut x=V::zeros();let mut previous=ll(x);let mut sum=V::zeros();
        for _ in 0..60000 {let (u,l,_)=latent_step(x,previous,mean,p,V::zeros(),proposal,ll,&mut rng);x=u;previous=l;sum+=x;}
        assert!((sum/60000.-truth).norm()<0.02);
    }
}
