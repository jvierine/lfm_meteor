//! Independent per-pulse physical delay/Doppler scatter, not receiver-noise inflation.
//! Exact sampled-voltage likelihood; local Gaussian fits are MH proposals only.
use super::*;
use super::compressed::RadiusTransport;
use nalgebra::{DMatrix, DVector, Matrix2, Vector2};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
use std::time::Instant;

type V = Vector2<f64>;
type M = Matrix2<f64>;

struct Datum {pulse:reduced::Pulse, origin:V, maximum:V, precision:M, offset:f64, station:usize}
struct Scatter {data:Vec<Datum>, path_scale:f64, beat_scale:f64}

struct Conditional {mean:V, factor:M, gaussian:f64}

fn conditional_gaussian(mean:V, maximum:V, measurement_cov:M, logs:[f64;2])->Option<Conditional> {
    let prior=M::from_diagonal(&V::new((2.*logs[0]).exp(),(2.*logs[1]).exp()));
    let sum=prior+measurement_cov;
    let inverse=sum.try_inverse()?;
    let covariance=if prior.trace()>measurement_cov.trace() {
        measurement_cov-measurement_cov*inverse*measurement_cov
    }else{prior-prior*inverse*prior};
    let covariance=(covariance+covariance.transpose())/2.;
    let factor=covariance.cholesky()?.l();
    let delta=maximum-mean;
    Some(Conditional{mean:mean+prior*inverse*delta,factor,
        gaussian:-0.5*delta.dot(&(inverse*delta))-0.5*(sum.determinant()/measurement_cov.determinant()).ln()})
}

fn shear()->M {M::new(1.,0.,1.,1.)}

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
    fn conditionals(&self,v:&Voltage,q:&[f64],logs:&[f64])->Option<Vec<Conditional>> {
        let means=self.means(v,q)?;let transform=shear();let inverse=M::new(1.,0.,-1.,1.);
        self.data.iter().enumerate().map(|(i,d)| {
            let precision=transform.transpose()*d.precision*transform;
            conditional_gaussian(inverse*means[i],inverse*d.maximum,precision.try_inverse()?,
                [logs[2*d.station],logs[2*d.station+1]])
        }).collect()
    }
    fn correction(&self,i:usize,c:&Conditional,epsilon:V)->f64 {
        let u=shear()*(c.mean+c.factor*epsilon);let d=&self.data[i];
        self.ll(i,u)-quadratic(u,d.maximum,d.precision)
    }
    fn proposal_metric(&self,v:&Voltage,q:&DVector<f64>,logs:&[f64],base:&DMatrix<f64>)->Option<DMatrix<f64>> {
        let inverse=M::new(1.,0.,-1.,1.);let transform=shear();
        let mut jacobians=vec![DMatrix::zeros(2,7);self.data.len()];
        for j in 0..7 {
            let step=if j==6 {1e-4}else{1e-5};
            let mut plus=q.clone();let mut minus=q.clone();plus[j]+=step;minus[j]-=step;
            let a=self.means(v,plus.as_slice())?;let b=self.means(v,minus.as_slice())?;
            for i in 0..a.len() {let derivative=inverse*(a[i]-b[i])/(2.*step);
                for k in 0..2 {jacobians[i][(k,j)]=derivative[k];}}
        }
        let mut information=DMatrix::zeros(13,13);
        for (i,d) in self.data.iter().enumerate() {
            let s=d.station;let measurement=(transform.transpose()*d.precision*transform).try_inverse()?;
            let variance=V::new((2.*logs[2*s]).exp(),(2.*logs[2*s+1]).exp());
            let weight=(measurement+M::from_diagonal(&variance)).try_inverse()?;
            let local=jacobians[i].transpose()*weight*&jacobians[i];
            for j in 0..7 {for k in 0..7 {information[(j,k)]+=local[(j,k)];}}
            for j in 0..2 {for k in 0..2 {
                information[(7+2*s+j,7+2*s+k)]+=2.*variance[j]*variance[k]*weight[(j,k)].powi(2);
            }}
        }
        for j in 0..6 {let t=(2.*logs[j]).exp()/3.;information[(7+j,7+j)]+=8.*t/(1.+t).powi(2);}
        // The ridge stabilizes the proposal metric only, never the posterior.
        let scale=base.clone().cholesky()?.l();
        let normalized=scale.transpose()*information*&scale+DMatrix::identity(13,13)*1e-8;
        let covariance=&scale*normalized.cholesky()?.inverse()*scale.transpose();
        Some((&covariance+covariance.transpose())/2.)
    }
}

#[no_mangle]
pub unsafe extern "C" fn voltage_scatter_sample(v:*const Voltage,stations:*const u64,q:*const f64,
    covariance:*const f64,warmup:usize,draws:usize,seconds:f64,seed:u64,path:*const c_char)->i32 {
    let result=(||->anyhow::Result<()> {
        let v=&*v;let q=std::slice::from_raw_parts(q,7);let stations=std::slice::from_raw_parts(stations,v.time.len());
        let prepared=Instant::now();let model=Scatter::build(v,q,stations)?;
        let mut rng=StdRng::seed_from_u64(seed);let x=DVector::from_column_slice(q);
        let base=DMatrix::from_row_slice(7,7,std::slice::from_raw_parts(covariance,49));
        let transport=RadiusTransport::new(&x,&base);
        let mut physical=transport.physical(&x);let means=model.means(v,physical.as_slice()).unwrap();
        let values:Vec<V>=model.data.iter().map(|d|d.maximum).collect();
        let mut logs=[-3.;6];
        for s in 0..3 {for j in 0..2 {
            let sum:Vec<f64>=model.data.iter().enumerate().filter(|(_,d)|d.station==s).map(|(i,_)| {
                let r=values[i]-means[i];if j==0 {r[0]}else{r[1]-r[0]}
            }).collect();
            logs[2*s+j]=(sum.iter().map(|r|r*r).sum::<f64>()/sum.len() as f64).sqrt().clamp(0.001,2.).ln();
        }}
        let mut state=DVector::zeros(13);
        state.rows_mut(0,7).copy_from(&x);
        for i in 0..6 {state[7+i]=logs[i];}
        let mut proposal_base=DMatrix::zeros(13,13);
        proposal_base.view_mut((0,0),(7,7)).copy_from(&(&base*100.));
        for i in 7..13 {proposal_base[(i,i)]=0.16;}
        let proposal_cov=model.proposal_metric(v,&physical,&logs,&proposal_base)
            .ok_or_else(||anyhow::anyhow!("Invalid Jacobian proposal metric"))?;
        let chol=proposal_cov.clone().cholesky().ok_or_else(||anyhow::anyhow!("Invalid proposal covariance"))?.l();
        let mut conditionals=model.conditionals(v,physical.as_slice(),&logs).ok_or_else(||anyhow::anyhow!("Invalid conditional covariance"))?;
        let mut epsilon=vec![V::zeros();values.len()];
        let mut corrections:Vec<f64>=(0..values.len()).map(|i|model.correction(i,&conditionals[i],epsilon[i])).collect();
        let mut scales=[0_f64;13];let mut component_trials=[0u64;13];let mut component_accepted=[0u64;13];
        let mut accepted=[0u64;5];
        let mut samples=Vec::new();let mut nuisance=Vec::new();let mut probability=Vec::new();
        let mut latent_sum=vec![V::zeros();values.len()];let mut latent_count=0usize;
        let elapsed_prepare=prepared.elapsed().as_secs_f64();let started=Instant::now();
        for step in 0..warmup+draws {
            if started.elapsed().as_secs_f64()>=seconds {break;}
            let component=step%13;
            let increment:f64=rng.sample(StandardNormal);
            let proposed=&state+chol.column(component)*(increment*scales[component].exp());
            let proposed_physical=transport.physical(&proposed.rows(0,7).into_owned());
            let proposed_logs=&proposed.as_slice()[7..13];
            let proposed_conditionals=model.conditionals(v,proposed_physical.as_slice(),proposed_logs);
            let mut proposed_corrections=Vec::new();
            let current=conditionals.iter().map(|c|c.gaussian).sum::<f64>()+corrections.iter().sum::<f64>()+nuisance_logprior(&logs);
            let candidate=if let Some(c)=&proposed_conditionals {
                proposed_corrections=(0..values.len()).map(|i|model.correction(i,&c[i],epsilon[i])).collect();
                c.iter().map(|x|x.gaussian).sum::<f64>()+proposed_corrections.iter().sum::<f64>()+nuisance_logprior(proposed_logs)
            }else{f64::NEG_INFINITY};
            let accept=candidate.is_finite() && rng.gen::<f64>().ln()<candidate-current;
            if accept {state=proposed;physical=proposed_physical;conditionals=proposed_conditionals.unwrap();
                corrections=proposed_corrections;logs.copy_from_slice(&state.as_slice()[7..13]);
                if step>=warmup {accepted[0]+=1;component_accepted[component]+=1;}}
            if step<warmup {
                let count=(step/13+10) as f64;
                scales[component]=(scales[component]+count.powf(-0.6)*((if accept {1.}else{0.})-0.44)).clamp(-8.,4.);
            }else{component_trials[component]+=1;}
            for i in 0..values.len() {
                let independent=rng.gen::<f64>()<0.75;
                let proposal=if independent {V::new(rng.sample(StandardNormal),rng.sample(StandardNormal))}
                    else {epsilon[i]+0.5*V::new(rng.sample(StandardNormal),rng.sample(StandardNormal))};
                let correction=model.correction(i,&conditionals[i],proposal);
                let gaussian=if independent {0.}else{-0.5*(proposal.norm_squared()-epsilon[i].norm_squared())};
                let ok=correction.is_finite() && rng.gen::<f64>().ln()<correction-corrections[i]+gaussian;
                if ok {epsilon[i]=proposal;corrections[i]=correction;if step>=warmup {accepted[4]+=1;}}
            }
            if step>=warmup {
                samples.extend_from_slice(physical.as_slice());
                for s in 0..3 {nuisance.push(logs[2*s].exp()*model.path_scale);nuisance.push(logs[2*s+1].exp()*model.beat_scale);}
                probability.push(conditionals.iter().map(|c|c.gaussian).sum::<f64>()+corrections.iter().sum::<f64>()
                    +nuisance_logprior(&logs)-0.5*epsilon.iter().map(|e|e.norm_squared()).sum::<f64>());
                if step%20==0 {for (i,sum) in latent_sum.iter_mut().enumerate(){*sum+=shear()*(conditionals[i].mean+conditionals[i].factor*epsilon[i]);}latent_count+=1;}
            }
        }
        let h=hdf5::File::create(CStr::from_ptr(path).to_str()?)?;
        h.new_dataset::<f64>().shape((probability.len(),7)).create("samples")?.write_raw(&samples)?;
        h.new_dataset::<f64>().shape((probability.len(),6)).create("scatter_samples")?.write_raw(&nuisance)?;
        h.new_dataset_builder().with_data(&probability).create("log_probability")?;
        h.new_dataset_builder().with_data(&accepted).create("accepted")?;
        h.new_dataset_builder().with_data(&component_trials).create("component_trials")?;
        h.new_dataset_builder().with_data(&component_accepted).create("component_accepted")?;
        h.new_dataset_builder().with_data(&scales.map(f64::exp)).create("frozen_component_scales")?;
        let metric:Vec<f64>=(0..13).flat_map(|i|(0..13).map(move |j|(i,j))).map(|(i,j)|proposal_cov[(i,j)]).collect();
        h.new_dataset::<f64>().shape((13,13)).create("jacobian_proposal_covariance")?.write_raw(&metric)?;
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
    fn conditional_coordinates_preserve_joint_density() {
        let mean=V::new(0.3,-0.2);let maximum=V::new(-0.1,0.7);
        let measurement=M::new(0.2,0.06,0.06,0.4);
        let precision=measurement.try_inverse().unwrap();
        let mut reference:Option<f64>=None;
        for logs in [[-6.,-5.],[-1.,0.3],[2.,3.]] {
            let c=conditional_gaussian(mean,maximum,measurement,logs).unwrap();
            let prior=M::from_diagonal(&V::new((-2.*logs[0]).exp(),(-2.*logs[1]).exp()));
            for epsilon in [V::zeros(),V::new(0.8,-1.2),V::new(-2.,0.4)] {
                let u=c.mean+c.factor*epsilon;
                let direct=quadratic(u,maximum,precision)+quadratic(u,mean,prior)
                    -logs[0]-logs[1]+c.factor.determinant().ln();
                let transformed=c.gaussian-0.5*epsilon.norm_squared();
                let difference=direct-transformed;
                if let Some(r)=reference {assert!((difference-r).abs()<1e-7);}
                else {reference=Some(difference);}
            }
        }
    }
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
