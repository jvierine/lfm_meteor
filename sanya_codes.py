import numpy as n
import scipy.constants as c
import matplotlib.pyplot as plt



def save_sanya_waveform_file(code,fname="sanya.bin",plot=True):
    """
        waveform in "code" has to be 30 MHz sample-rate
        16-bit integer I
        16-bit integer Q
        sampler-rate 30 MHz
        2^16 samples data length
        amplitude = 2^15 - 1
    """
    I=n.zeros(2**16,dtype=n.int16)
    Q=n.zeros(2**16,dtype=n.int16)
    I[0:len(code)]=n.floor(n.real(code)*(2**15-1))
    Q[0:len(code)]=n.floor(n.imag(code)*(2**15-1))
    if plot:
        plt.plot(I)
        plt.plot(Q)
        plt.show()
    # first store I
    f=open(fname,"wb")
    I.tofile(f)
    Q.tofile(f)
    f.close()
    print("wrote %s"%(fname))

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
    om=bw*1e6/l/2.0
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
    plt.pcolormesh(dops/1e3,rgs,dB,vmin=mdb-6,vmax=mdb)
    plt.title("Range-Doppler ambiguity function")#\n N=199 samples sr=4 MHz B=4 MHz radar_freq=440 MHz")
    cb=plt.colorbar()
    cb.set_label("dB")
    plt.xlabel("Doppler (km/s)")
    plt.ylabel("Range (m)")    
    plt.show()

if __name__ == "__main__":
    code1=lfm(l=200,sr=30)
    range_doppler_ambiguity(code1, sr=30e6, dops=n.linspace(-100e3,100e3,num=300) )


    code1=lfm(l=100,sr=30)
    #code1=lfm(l=100)
    code2=n.concatenate((code1,n.conj(code1[::-1])))
    # save 4 MHz up-down chirp
    save_sanya_waveform_file(code2,fname="up_down_lfm_4MHz.bin")
    tvec=1e6*n.arange(len(code2))/30e6

    plt.plot(tvec,code2.real*(2**15-1))
    plt.plot(tvec,code2.imag*(2**15-1))
    plt.plot(tvec,n.abs(code2*(2**15-1)))

    plt.xlabel("Time ($\mu$s)")
    plt.title("4 MHz Up-Down LFM Chirp")
    plt.show()
    range_doppler_ambiguity(code2, sr=30e6, dops=n.linspace(-100e3,100e3,num=300) )

    code1=lfm(l=100,sr=30,bw=16e6)
    code2=n.concatenate((code1,n.conj(code1[::-1])))
    # save 16 MHz up-down chirp
    save_sanya_waveform_file(code2,fname="up_down_lfm_16MHz.bin")

    plt.plot(tvec,code2.real*(2**15-1))
    plt.plot(tvec,code2.imag*(2**15-1))
    plt.plot(tvec,n.abs(code2*(2**15-1)))
    plt.xlabel("Time ($\mu$s)")
    plt.title("16 MHz Up-Down LFM Chirp")
    plt.show()


    range_doppler_ambiguity(code2, sr=30e6, dops=n.linspace(-100e3,100e3,num=300), nint=4 )
 #   range_doppler_ambiguity(code2,sr=10, dops=n.linspace(-1e3,1e3,num=300),ranges=100)

  #  range_doppler_ambiguity(code2 )

