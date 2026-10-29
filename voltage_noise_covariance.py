"""Estimate proper-complex AR noise from contiguous off-echo voltage only."""
import numpy as np
from scipy.signal import lfilter


def fit_noise(rows, gates, max_order=16):
    segments=[]
    for row, gate in zip(rows,gates):
        for segment in (row[:max(0,int(gate)-650)],row[min(len(row),int(gate)+650):]):
            if len(segment)>=128:
                segments.append(np.asarray(segment,dtype=complex))
    if len(segments)<6:
        raise ValueError('Insufficient independent off-echo segments for covariance estimation')
    train=[s for i,s in enumerate(segments) if i%4]
    test=[s for i,s in enumerate(segments) if not i%4]
    order_max=max_order
    # Each regression uses only contiguous lags, never bridges the echo mask.
    def matrices(parts):
        xs=[];ys=[]
        for s in parts:
            win=np.lib.stride_tricks.sliding_window_view(s,order_max+1)
            ys.append(win[:,-1]);xs.append(win[:,:-1][:,::-1])
        return np.concatenate(xs),np.concatenate(ys)
    x,y=matrices(train);xt,yt=matrices(test)
    candidates=[]
    for p in (0,1,2,4,8,16):
        if p>max_order:continue
        coeff=np.linalg.lstsq(x[:,:p],y,rcond=None)[0] if p else np.empty(0,complex)
        residual=y-x[:,:p]@coeff
        variance=np.mean(abs(residual)**2)
        bic=2*len(y)*np.log(variance)+2*p*np.log(len(y))
        if p and np.max(abs(np.roots(np.r_[1.,-coeff])))>=.9999:
            continue
        candidates.append((bic,coeff,variance))
    _,coeff,variance=min(candidates,key=lambda item:item[0])
    whiten=np.r_[1.,-coeff]
    residuals=[lfilter(whiten,[1.],s)[order_max:] for s in test]
    z=np.concatenate(residuals)
    correlations=[]
    for lag in range(1,33):
        numerator=sum(np.vdot(s[:-lag],s[lag:]) for s in residuals)
        denominator=sum(np.sum(abs(s[:-lag])**2) for s in residuals)
        correlations.append(numerator/denominator)
    # Broad familywise diagnostic; these are flags, not a fitted variance inflation.
    tolerance=5/np.sqrt(len(z))
    pseudo=np.mean(z*z)/np.mean(abs(z)**2)
    mean_z=abs(np.mean(z))/np.sqrt(np.mean(abs(z)**2)/len(z))
    numerator=0j;den1=0.;den2=0.;pairs=0
    for i in range(len(rows)-1):
        idx=np.arange(rows.shape[1]);mask=(abs(idx-gates[i])>650)&(abs(idx-gates[i+1])>650)
        a,b=rows[i,mask],rows[i+1,mask]
        numerator+=np.vdot(a,b);den1+=np.vdot(a,a).real;den2+=np.vdot(b,b).real;pairs+=len(a)
    temporal=numerator/np.sqrt(den1*den2) if den1*den2>0 else complex(np.nan)
    temporal_pass=bool(pairs>0 and abs(temporal)<5/np.sqrt(pairs))
    whiteness=bool(np.max(abs(np.array(correlations)))<tolerance and abs(pseudo)<tolerance and mean_z<5 and temporal_pass)
    return dict(coeff=coeff,whitener=whiten,innovation_variance=variance,
                heldout_variance=float(np.mean(abs(z)**2)),heldout_correlation=np.array(correlations),
                heldout_pseudocorrelation=pseudo,heldout_count=len(z),whiteness_pass=whiteness,
                heldout_tolerance=tolerance,order=len(coeff),mean_z=mean_z,
                temporal_correlation=temporal,temporal_pass=temporal_pass)


def colored_noise(rng, n, coeff, innovation_variance):
    coeff=np.asarray(coeff,dtype=complex)
    burn=max(256,16*len(coeff))
    z=np.sqrt(innovation_variance/2)*(rng.normal(size=n+burn)+1j*rng.normal(size=n+burn))
    return lfilter([1.],np.r_[1.,-coeff],z)[burn:]
