import glob
import h5py
import numpy as n
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


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
    

def readsanya(f):

    code=lfm()
    if False:
        plt.specgram(code,NFFT=64,Fs=4e6,noverlap=16)
        plt.show()
    
    if False:
        plt.plot(code.real)
        plt.plot(code.imag)
        plt.show()

    h=h5py.File(f,"r")
    zz=h["data_raw"][()]
    print(zz["real"].shape)
    print(zz["imag"].shape)
#    exit(0)
    p=h["para"][()]
    tm=h["time"][()]
    print(tm)
    print(tm[:,0])
    print(tm.shape)
    az=p[6]
    el=p[7]
    t_p=p[10]
    t_ipp=p[11]
    r0=p[12]
    r1=p[13]
    sr=p[14]
    bw=p[15]
#    plt.plot(tm[5,:])
 #   plt.show()
    print("az %1.2f (deg) el %1.2f (deg) tp %1.2f (us) tipp %1.2f (us) r0 %1.2f (km) r1 %1.2f (km) sr %1.2f (MHz) bw %1.2f (MHz)"%(az,el,t_p,t_ipp,r0,r1,sr,bw))

    #There are three variables in each mat file.
    #1. para: The experimental configuration parameters. Maybe, you only need the para([ 7 8 10 12 13 14 15 16]) . There are the azimuth (degree),
    #elevation (degree), pulse width of linear frequency modulation (us), IPP (us), gate start (km), gate end (km), sampling rate(MHz),
    #and band width(MHz).
    #2. time: The local time of Beijing (UTC+8), dimension is Nx7 with [year,month,day,hour,minute,second,code], here all codes are 0, you do not need.
    #3. data_raw: The raw IQ data with dimensions [time, sampling points at range].
    #Range=para(13):0.15/para(15):para(13)+(size(data_raw,2)-1)*0.15/para(15).

    #The site geographic information in the WGS84 :
    #lat0  = [18.3492  ; 19.5281  ; 19.5982  ];     % Lat.  of Sanya, Danzhou, Wenchang
    #lon0 = [109.6222; 109.1322; 110.7908];     % Lon. of Sanya, Danzhou, Wenchang
    #alt0  = [0.05        ; 0.0999    ;  0.0249   ];     % Alt.   of Sanya, Danzhou, Wenchang
    
    h.close()
    z=n.array(zz["real"]+zz["imag"]*1j,dtype=n.complex64)
    ranges=r0+3e8*n.arange(len(z))/(1e6*sr)/2/1e3
    zd=n.copy(z)
    print(z.shape)
    L = n.conj(n.fft.fft(code,4096))
    echoes=[]
    raw=[]
    dts=[]
    for i in range(z.shape[1]):
        zd[:,i]=n.convolve(z[:,i],n.conj(code),mode="same")#n.fft.ifft(n.fft.fft(z[:,i],4096)*L)[0:zd.shape[0]]
        noise=n.median(n.abs(zd[:,i]))
        if n.max(n.abs(zd[:,i]))/noise > 10:
            base_dt = n.datetime64(f'{int((tm[0,i]+2000)):04d}-{int(tm[1,i]):02d}-{int(tm[2,i]):02d}T{int(tm[3,i]):02d}:{int(tm[4,i]):02d}')
            dt=base_dt + n.timedelta64(int(n.floor(tm[5,i])),"s")+n.timedelta64(int(n.round(1e9*(tm[5,i]-int(n.floor(tm[5,i]))))),"ns")
            print(tm[0,i])
            print(dt)
            dts.append(dt)
            echoes.append(zd[:,i])
            raw.append(z[:,i])
  #          plt.specgram(z[:,i],NFFT=64,Fs=4e6,noverlap=16)
   #         plt.show()
#            plt.specgram(code,NFFT=64,Fs=4e6,noverlap=16)
 #           plt.show()
            if False:
                plt.subplot(121)
                plt.plot(z[:,i].real)
                plt.plot(z[:,i].imag)
                plt.subplot(122)
                plt.plot(n.abs(zd[:,i]))
                plt.show()

    times=n.arange(len(echoes))*t_ipp/1e6
    if len(echoes)>2:
        echoes=n.array(echoes)
        raw=n.array(raw)

        plt.pcolormesh(dts,ranges,n.abs(echoes.T))
        #
        plt.title(dts[0])
        plt.savefig("sanya%02d-%02d-%02d %02d-%02d-%1.2f.png"%(tm[0,0],tm[1,0],tm[2,0],tm[3,0],tm[4,0],tm[5,0]))
        plt.close()
#        plt.show()
#        plt.pcolormesh(times,n.arange(echoes.shape[1]),n.abs(raw.T))
 #       plt.show()


dirnames=["/data1/SANYA/Juha/20240422/Sanya",
          "/data1/SANYA/Juha/20240422/Wenchang",
          "/data1/SANYA/Juha/20240422/Danzhou"]

fls=[]
for d in dirnames:
    fl=glob.glob("%s/*.mat"%(d))
    fl.sort()
    fls.append(fl)

for fi in range(rank,len(fls[0]),size):
    f0=fls[0][fi]
    f1=fls[1][fi]
    f2=fls[2][fi]
    print(f0)
    print(f1)
    print(f2)
    readsanya(f0)
 #   readsanya(f1)
  #  readsanya(f2)        

