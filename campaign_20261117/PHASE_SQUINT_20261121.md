# 21 November fixed-phase steering hypothesis

The available binary headers contain commanded directions, not element steering
weights. The 440 MHz fixed-phase design is an assumption supplied by the user;
the tables/controller were unavailable. This calculation therefore predicts a
pointing shift and is not an independent calibration. True time delays or
frequency-adjusted steering would invalidate this correction.

Using the array-plane direction-cosine correction at430MHz:

| Station | Header az/el (deg) | Predicted az/el (deg) |
|---|---|---|
| Sanya |14.996338 /74.998169|14.996338 /74.640791|
| Danzhou |149.979858 /42.099609|149.879764 /41.385846|
| Wenchang |226.499634 /33.596191|226.553945 /32.605141|

WGS84 ray geometry and the existing remote rectangular-aperture model give:

| Steering scenario | Danzhou peak (km) | Wenchang peak (km) | Joint gain peak (km) | Shared half-power interval (km) |
|---|---:|---:|---:|---|
| Header all arrays |109.978|109.983|109.98|107.99–111.98|
| Remote correction only |107.810|106.568|107.31|105.83–108.98|
| All arrays corrected |107.298|106.153|106.84|105.34–108.55|

The final prediction overlaps the105–108km observed reference band. This is
consistency, not identification of the hardware steering mode or validation of
the approximate gain pattern. The remote model uses32m×24m apertures and the
existing mechanical normals. Sanya uses the40m horizontal aperture.

## Sanya gain sensitivity kept separate

At106.5km, the relative one-way−3dB horizontal projected collecting areas are
2.4663km² (header),2.4794km² (phase prediction), and2.4819km² (empirical axis).
The phase prediction changes this contour area by+0.53%. At a fixed absolute
gain threshold, the corresponding areas are2.4663,2.4675,2.4681km²; that is
+0.05% for the phase prediction. These contour areas are not a complete
population/detection-weighted flux collecting area.

RCS sensitivity is much larger for individual frozen target directions:
using the predicted axis instead of the header axis multiplies inferred
monostatic RCS by2.276 at the old header centre, or by0.447 at the predicted
centre. These are location-specific ratios, not a catalogue-wide multiplier.
The empirical Sanya axis14.737741/74.618191deg is an alternative: do not add
the phase correction to it. No existing RCS/flux product was overwritten.
Timing, trajectories, masses and their fits remain unchanged.

The HDF5 product retains header angles, scenario directions, signed coordinate
conventions, gain profiles, peak heights, area definitions and input hashes.
