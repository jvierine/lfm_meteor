import numpy as n
import scipy.constants as c
import matplotlib.pyplot as plt
import jcoord

lat0  = n.array([18.3492,19.5281,19.5982  ])#;     % Lat.  of Sanya, Danzhou, Wenchang
lon0 = n.array([109.6222,109.1322,110.7908])#;     % Lon. of Sanya, Danzhou, Wenchang
alt0  = n.array([0.05     ,0.0999 ,0.0249   ])#;     % Alt.   of Sanya, Danzhou, Wenchang

p_san=jcoord.geodetic2ecef(lat0[0], lon0[0], alt0[0]*1e3)
p_dan=jcoord.geodetic2ecef(lat0[1], lon0[1], alt0[1]*1e3)
p_wen=jcoord.geodetic2ecef(lat0[2], lon0[2], alt0[2]*1e3)


def lfm(l=199,sr=4,bw=4e6):
    tidx=n.arange(l*sr)/(sr*1e6)
    #phi = o*t**2.0
    #f = 2*o*t
    # df/dt = 2*o
    # bw*1e6
    # delta t = l/1e6
    # df = bw
    # o = (delta f) / (2*delta t)
    #
    om = bw / (2.0*(l/1e6))

    #2*om*199/1e6 = bw
    om=bw*1e6/199/2.0
    # positive to negative LFM
    return(n.array(n.exp(1j*2*n.pi*(tidx*bw/2-om*tidx**2.0)),dtype=n.complex64))

def range_doppler_ambiguity(code,dops=n.linspace(-100e3,100e3,num=300),ranges=100,sr=4e6,freq=440e6,nint=1):
    padded=n.concatenate((n.repeat(n.zeros(ranges,dtype=n.complex64),nint),n.repeat(code,nint),n.repeat(n.zeros(ranges,dtype=n.complex64),nint)))
    rgs=c.c*(n.arange(2*nint*ranges)-nint*ranges)/(nint*sr)/2

    nrg=len(rgs)
    print(nrg)
    ndops=len(dops)
    S=n.zeros([nrg,ndops],dtype=n.float32)
    idx=n.arange(nint*len(code),dtype=int)
    cc=n.conj(n.repeat(code,nint))
    dopf=2*freq*dops/c.c
    for i in range(nrg):
#        print(i)
        for j in range(ndops):
            dopc=n.exp(1j*2*n.pi*dopf[j]*idx/(nint*sr))
            S[i,j]=n.abs(n.sum(dopc*padded[idx+i]*cc))**2.0
    S=S/n.max(S)
    dB=10.0*n.log10(S)
    mdb=n.max(dB)
    plt.pcolormesh(dops/1e3,rgs,dB,vmin=mdb-2,vmax=mdb)
    plt.title("Range-Doppler ambiguity function\n N=199 samples sr=4 MHz B=4 MHz radar_freq=440 MHz")
    plt.colorbar()
    plt.xlabel("Doppler (km/s)")
    plt.ylabel("Range (m)")    
    plt.show()

if __name__ == "__main__":
    code=lfm()
    range_doppler_ambiguity(code )
    
