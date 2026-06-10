import jcoord 
import numpy as n

lat0  = n.array([18.3492,19.5281,19.5982  ])#;     % Lat.  of Sanya, Danzhou, Wenchang
lon0 = n.array([109.6222,109.1322,110.7908])#;     % Lon. of Sanya, Danzhou, Wenchang
alt0  = n.array([0.05     ,0.0999 ,0.0249   ])#;     % Alt.   of Sanya, Danzhou, Wenchang

p_san=jcoord.geodetic2ecef(lat0[0], lon0[0], alt0[0]*1e3)
p_dan=jcoord.geodetic2ecef(lat0[1], lon0[1], alt0[1]*1e3)
p_wen=jcoord.geodetic2ecef(lat0[2], lon0[2], alt0[2]*1e3)