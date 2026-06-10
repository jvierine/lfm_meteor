import h5py
import glob
import scipy.constants as c

fl=glob.glob("/mnt/data/juha/SANYA/Juha/20240422/Sanya/*.mat")
h=h5py.File(fl[0],"r")
#print(h.keys())
#print(h["para"][()])



p=h["para"][()]
#for i in range(len(p)):
 #   print("%d %1.2f"%(i+1,h["para"][()][i]))


rt=h["para"][()][12,0]
rc=rt

# timesampe of first sample
t0 = (1e3*(rt+rc))/c.c
print("Sanya time of first sample %1.2f (microseconds)"%(t0*1e6))
h.close()
fl=glob.glob("/mnt/data/juha/SANYA/Juha/20240422/Danzhou/*.mat")
h=h5py.File(fl[0],"r")

# The first range is
rt=h["para"][()][12,0]

# time of first sample
# 
t0 = (69.9e3+28.1378e3)/(c.c/2)
print("Danzhou time of first sample %1.2f (microseconds)"%(t0*1e6))



h.close()
fl=glob.glob("/mnt/data/juha/SANYA/Juha/20240422/Wenchang/*.mat")
h=h5py.File(fl[0],"r")

# The first range is
rt=h["para"][()][12,0]

# time of first sample
t0 = (69.9e3+45.7489e3)/(c.c/2)
print("Wenchang time of first sample %1.2f (microseconds)"%(t0*1e6))

h.close()


