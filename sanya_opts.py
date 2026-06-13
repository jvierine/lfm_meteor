import jcoord 
import numpy as n

lat0  = n.array([18.3492,19.5281,19.5982  ])#;     % Lat.  of Sanya, Danzhou, Wenchang
lon0 = n.array([109.6222,109.1322,110.7908])#;     % Lon. of Sanya, Danzhou, Wenchang
alt0  = n.array([0.05     ,0.0999 ,0.0249   ])#;     % Alt.   of Sanya, Danzhou, Wenchang

p_san=jcoord.geodetic2ecef(lat0[0], lon0[0], alt0[0]*1e3)
p_dan=jcoord.geodetic2ecef(lat0[1], lon0[1], alt0[1]*1e3)
p_wen=jcoord.geodetic2ecef(lat0[2], lon0[2], alt0[2]*1e3)

# Satellite TLE calibration, 2024-04-22 Sanya data:
# range_offset = observed Sanya range - TLE-predicted aliased range.
SANYA_TLE_RANGE_OFFSET_KM = 16.0186
SANYA_RANGE_CORRECTION_KM = -SANYA_TLE_RANGE_OFFSET_KM

# First-sample delay calibration used for the corrected tri-static meteor
# geometry and trajectory fits.  The Sanya value is the raw transmit-target
# range origin; the range correction above is applied separately to the Sanya
# one-way range.  The remote values are the rounded receiver delays from the
# Memo 3 beam-axis calibration, where 1 us accuracy is sufficient.
SANYA_FIRST_SAMPLE_DELAY_US = 466.320
DANZHOU_FIRST_SAMPLE_DELAY_US = 359.0
WENCHANG_FIRST_SAMPLE_DELAY_US = 360.0

SANYA_TLE_RANGE_OFFSET_PREFILTER_CENTER_KM = 16.0
SANYA_TLE_RANGE_OFFSET_PREFILTER_HALF_WIDTH_KM = 2.0
SANYA_TLE_RANGE_OFFSET_PREFILTER_MAX_BEAM_ANGLE_DEG = 2.0
