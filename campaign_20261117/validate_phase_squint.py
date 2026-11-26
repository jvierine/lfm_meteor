"""Conditional fixed-phase 440/430 MHz pointing and frozen-geometry gain audit.

No timing, trajectory, empirical pointing, RCS, or flux product is overwritten.
"""
import argparse,hashlib
from pathlib import Path
import h5py,numpy as np
from scipy.optimize import minimize_scalar,brentq
from astropy.coordinates import EarthLocation
from astropy import units as u
from analyze import STATIONS,ENU,LLH,C,enu_matrix,save
from derived import phase_table_pointing,beam_frame,beam,beam_power,RCS_AZ_EL
BORES=[(0.,90.),(158.3,70.),(221.9,70.)]
SITES=['sanya','danzhou','wenchang']
def points_at_height(heights,pointing):
 axis=ENU.T@beam_frame(*pointing)[0]
 heights=np.atleast_1d(heights);r=(heights*1000-LLH[0,2])/axis.dot(ENU[2])
 for _ in range(7):
  xyz=STATIONS[0]+r[:,None]*axis
  z=EarthLocation.from_geocentric(*[xyz[:,j]*u.m for j in range(3)]).height.to_value(u.m)
  r+=(heights*1000-z)/(axis@ENU[2])
 np.testing.assert_allclose(z,heights*1000,atol=1e-5)
 return xyz

def remote_gain(xyz,index,pointing):
 normal=beam_frame(*BORES[index])[0];axis=beam_frame(*pointing)[0]
 tilt=np.array([0.,0.,1.])-normal[2]*normal;tilt/=np.linalg.norm(tilt);cross=np.cross(normal,tilt)
 d=(xyz-STATIONS[index])@enu_matrix(*LLH[index,:2]).T;d/=np.linalg.norm(d,axis=-1,keepdims=True)
 power=np.sinc(32/(C/430e6)*((d-axis)@tilt))**2*np.sinc(24/(C/430e6)*((d-axis)@cross))**2
 return 43+25*np.log10(normal@axis)+10*np.log10(np.maximum(power,1e-30))

def area(altitude,pointing,cutoff=-3.,step=.025,absolute_threshold_db=None):
 # WGS84 constant-geodetic-height surface in the Sanya tangent chart.
 axis=beam_frame(*pointing)[0];up=altitude-LLH[0,2]/1000;center=axis[:2]/axis[2]*up
 co=np.arange(-8.,8.+step/2,step);x,y=np.meshgrid(co+center[0],co+center[1]);z=np.full(x.shape,up*1000)
 for _ in range(5):
  local=np.stack([x*1000,y*1000,z],-1);xyz=STATIONS[0]+local@ENU
  h=EarthLocation.from_geocentric(*[xyz[...,j]*u.m for j in range(3)]).height.to_value(u.m);z+=altitude*1000-h
 d=np.stack([x*1000,y*1000,z],-1);d/=np.linalg.norm(d,axis=-1,keepdims=True)
 relative=10*np.log10(np.maximum(beam_power(d,axis),1e-30))
 if absolute_threshold_db is not None:relative+=46+25*np.log10(np.sin(np.deg2rad(pointing[1])))
 inside=relative>=(cutoff if absolute_threshold_db is None else absolute_threshold_db)
 assert not any([inside[0].any(),inside[-1].any(),inside[:,0].any(),inside[:,-1].any()])
 return inside.sum()*step**2

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('headers',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 with h5py.File(a.headers) as h:
  commanded=np.array([[h[s]['rx_azimuth_deg'][0],h[s]['rx_elevation_deg'][0]] for s in SITES])
  np.testing.assert_array_equal(commanded[0],[h['sanya_late']['rx_azimuth_deg'][0],h['sanya_late']['rx_elevation_deg'][0]])
 corrected=np.array([phase_table_pointing(q,b) for q,b in zip(commanded,BORES)])
 expected=np.array([[14.996,74.641],[149.880,41.386],[226.554,32.605]])
 np.testing.assert_allclose(corrected,expected,atol=.001)
 # Independent phase-gradient check on a planar aperture at the new frequency.
 for cmd,bore,new in zip(commanded,BORES,corrected):
  n=beam_frame(*bore)[0];v=beam_frame(*cmd)[0];w=beam_frame(*new)[0]
  np.testing.assert_allclose(430*(w-(w@n)*n),440*(v-(v@n)*n),atol=1e-11)
  np.testing.assert_allclose(phase_table_pointing(cmd,bore,430,430),cmd,atol=1e-10)
  np.testing.assert_allclose(phase_table_pointing(new,bore,430,440),cmd,atol=1e-10)
 heights=np.linspace(90,120,3001);gains=[];peaks=[];bands=[];overlap_peaks=[]
 settings=[commanded,np.vstack([commanded[0],corrected[1:]]),corrected]
 for angles in settings:
  xyz=points_at_height(heights,angles[0]);g=np.array([remote_gain(xyz,s,angles[s]) for s in (1,2)]);gains.append(g)
  peak=[]
  for j in range(2):
   k=np.argmax(g[j]);lo=heights[max(0,k-2)];hi=heights[min(len(heights)-1,k+2)]
   opt=minimize_scalar(lambda z:-remote_gain(points_at_height(z,angles[0]),j+1,angles[j+1])[0],bounds=(lo,hi),method='bounded');peak.append(opt.x)
  peaks.append(peak)
  keep=np.all(g>=np.max(g,axis=1)[:,None]-3,axis=0);bands.append([heights[keep].min(),heights[keep].max()] if keep.any() else [np.nan,np.nan])
  overlap_peaks.append(heights[np.argmax(g.sum(axis=0))])
 areas=[];fixedareas=[]
 threshold=46+25*np.log10(np.sin(np.deg2rad(commanded[0,1])))-3
 for q in [commanded[0],corrected[0],np.array(RCS_AZ_EL)]:
  coarse=area(106.5,q,step=.05);fine=area(106.5,q);assert abs(coarse/fine-1)<.02
  areas.append(fine);fixedareas.append(area(106.5,q,absolute_threshold_db=threshold))
 # Frozen physical locations: three alternative axes, never composed corrections.
 reference_points=np.concatenate([points_at_height([105,106.5,108],q) for q in [commanded[0],corrected[0],RCS_AZ_EL]])
 gains_sanya=np.array([beam(reference_points,*q)[1] for q in [commanded[0],corrected[0],RCS_AZ_EL]])
 save(a.output,dict(commanded_az_el_deg=commanded,predicted_az_el_deg=corrected,mechanical_boresight_az_el_deg=BORES,altitude_km=heights,remote_gain_dbi=np.array(gains),remote_peak_height_km=peaks,joint_half_power_interval_km=bands,remote_gain_product_peak_height_km=overlap_peaks,collecting_area_km2=areas,fixed_absolute_gain_threshold_area_km2=fixedareas,reference_position_ecef_m=reference_points,sanya_absolute_gain=gains_sanya,rcs_ratio_phase_vs_header=(gains_sanya[0]/gains_sanya[1])**2,rcs_ratio_phase_vs_empirical=(gains_sanya[2]/gains_sanya[1])**2),dict(fixed_phase_status='assumed; actual steering tables/controller unavailable',pointing_status='model prediction, not independent calibration',scenario_order='header_all,remote_phase_only,all_phase',area_axis_order='header,phase_prediction,empirical_alternative',area_definition='horizontal projected area on WGS84 geodetic106.5km surface; relative one-way -3dB contour; not absolute flux effective area',fixed_gain_threshold_dbi=threshold,rcs_ratio_definition='sigma_phase/sigma_other=(G_other/G_phase)^2 for fixed measured power and position',empirical_alternative_az_el_deg=RCS_AZ_EL,timing_and_trajectory='unchanged; no fits performed',header_sha256=hashlib.sha256(a.headers.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 print('ANGLES',corrected);print('PEAKS',peaks);print('OVERLAP',bands,overlap_peaks);print('AREA',areas,'FIXED',fixedareas);print('RCS_RATIO', (gains_sanya[0]/gains_sanya[1])**2)
if __name__=='__main__':main()
