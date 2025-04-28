import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob
import stuffr
import jcoord
import sanya_codes as sc
import scipy.optimize as so

#(obs_lat, obs_lon, obs_h, az, el, r):
# initial guess position
pguess_llh=jcoord.az_el_r2geodetic(sc.lat0[0], sc.lon0[0], sc.alt0[0]*1e3, 15, 75,100e3)
print(pguess_llh)
pguess=jcoord.geodetic2ecef(pguess_llh[0],pguess_llh[1],pguess_llh[2])

def triangulate(sd,dd,wd):
    def ss(x):
        m_sd=n.linalg.norm(x-sc.p_san)
        m_dd=n.linalg.norm(x-sc.p_dan)
        m_wd=n.linalg.norm(x-sc.p_wen)
        sq=(m_sd-sd)**2.0+(m_dd-dd)**2.0+(m_wd-wd)**2.0
        print(sq)
        return(sq)
    xhat=so.fmin(ss,pguess)
    return(xhat)

def get_ev(pref="san"):
    fl=glob.glob("%s*.h5"%(pref))
    fl.sort()
    ev=[]
    for f in fl:
        h=h5py.File(f,"r")
#        print(h["az"][()])
 #       print(h["el"][()])        
        echoes=h["echoes"][()]    
        rgs=[]
        for i in range(echoes.shape[0]):
            rgs.append(n.argmax(n.abs(echoes[i,:])))
        ev.append({"t":h["times_ns"][()],"t0":n.min(h["times_ns"][()]),"t1":n.max(h["times_ns"][()]),"rgs":rgs})
        h.close()
    return(ev)

sev=get_ev("san")
wev=get_ev("w")
dev=get_ev("d")

for e in sev:
    e["d"]=None
    e["w"]=None    
    for de in dev:
        # starts during s
        if (de["t0"] > e["t0"]) and (de["t0"] < e["t1"]):
            print("d")
            e["d"]=de
        # end during s
        if (de["t1"] > e["t0"]) and (de["t1"] < e["t1"]):
            print("d")            
            e["d"]=de
    for we in wev:
        # starts during s
        if (we["t0"] > e["t0"]) and (we["t0"] < e["t1"]):
            print("w")            
            e["w"]=we
        # end during s
        if (we["t1"] > e["t0"]) and (we["t1"] < e["t1"]):
            print("w")            
            e["w"]=we

    if e["d"] != None and e["w"] != None:
        e["tri"]=True
        plt.plot((e["t"]-n.min(e["t"]))/1e9,e["rgs"],".",label="Sanya")
        plt.plot((e["d"]["t"]-n.min(e["t"]))/1e9,e["d"]["rgs"],".",label="Danzhou")
        plt.plot((e["w"]["t"]-n.min(e["t"]))/1e9,e["w"]["rgs"],".",label="Wenchang")
        plt.ylabel("Range gate")
        plt.xlabel("Time (s)")
        plt.title("%s"%(stuffr.unix2datestr(n.min(e["t"])/1e9)))
        plt.legend()
        plt.savefig("tri-%d.png"%(n.min(e["t"])))
        plt.close()
#        plt.show()

    

        
    t0=e["t0"]
    t0=e["t1"]    
