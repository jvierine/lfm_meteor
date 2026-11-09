Sanya Tri-static Meteor Head Echo Experiment
Reviewer release: 2026-11-09
Reserved DOI: 10.5281/zenodo.24037714

This is an unpublished reviewer draft. Public release is planned upon paper
acceptance. Extract the three archives into the same empty directory.

FILES
sanya-raw-data.tar.gz: unchanged event-only voltages and detector outputs
  (5,215 station event cuts). No continuous raw recording is included.
sanya-results-20261109.tar.gz: current trajectories, manual exclusions,
  calibrated dynamic-mass confidence bounds, RCS measurements and mass losses,
  lookup table, orbit ensembles, and the numerical products used in the paper.
sanya-code-20261109.tar.gz: processing source and a release-verification script.
read_sanya_zenodo.py: standalone reader for event curves and orbital elements.
checksums-20261109.h5: file hashes and sizes. The results archive also contains
  a member-level HDF5 inventory. The code archive has its own inventory.

CURRENT PRODUCTS
L1/events/: 234 retained direct shrinking-radius trajectory fits, fitted
  positions and velocity vectors, SNR/RCS, and diagnostic curves. The event
  input.h5 contains the actual masked voltage samples used in the fit.
  mass_confidence.h5, where available, contains the calibrated dynamic-mass
  bounds. Missing or unbounded intervals are not replaced with old intervals.
  Historical posterior mass arrays have been removed from the event exports.
  The stored trajectory bands remain conditional posterior bands, not
  calibrated joint mass-confidence projections.
results/tristatic_calibrated_likelihood_20261109/:
  mass_selection.h5 is the authoritative dynamic-mass selection for all
  234 retained events: 135 selected, 86 broad/unbounded, and 13 numerically
  unresolved. An unresolved numerical calculation is not evidence of a
  mathematically unbounded interval. No Whipple-Jacchia fallback is used.
  confidence/ contains event likelihood profiles and calibrated bounds;
  bootstrap/ contains pulse-multiplier optimizations and calibration results;
  inputs/ and flags_snapshot/ freeze their measurement and exclusion inputs.
  products/mass_distribution.h5 contains 135 selected dynamic masses and
  115 quality-selected pairs with RCS mass loss.
  current_mass_influx.h5 retains all 197 RCS mass losses and their draws.
  products/imputed_mass_influx.h5 contains the reported population estimates.
  For events without usable mass bounds, the median corrected contribution
  of the selected events is assigned, not a claimed measured physical mass.
  The sensitivity ranges include one common velocity exponent 3 <= p <= 3.5:
  33.744--169.133 tonnes/day (dynamic) and 2.542--15.727 tonnes/day (RCS).
  These are not calibrated global 95% confidence intervals. Other physical,
  calibration, detection-correction and missing-mass uncertainties are excluded.
results/tristatic_seven_parameter_catalogue_20261108/products/:
  trajectory, radiant, noise, beam, altitude-selection and population inputs.
  Superseded dynamic-mass products are not included in this directory.
orbits/: 216 verified orbital ensembles, 100 samples per event, propagated
  1000 days backward. All 21,600 retained samples are bound. Eighteen retained
  trajectories lack verified orbital ensembles and are not silently filled.
uncertainty/trajectory/: conditional trajectory posterior samples and diagnostics;
  these are not the dynamic-mass uncertainty estimator used in the paper.
lookup/: SYISR RCS lookup used for the RCS-derived mass losses, with its
  existing interpolation, low-speed override and clamping metadata retained.
support/original_inputs/: full pulse lists, including points excluded from fits,
  needed to reconstruct diagnostics and nominal-time state-vector transforms.

PHYSICAL ASSUMPTIONS
Dynamic mass is mass at first detection. RCS mass is mass ablated during the
observed beam crossing, within 10 dB of the one-way Sanya gain peak. It is not
the mass above the atmosphere. Both exclude earlier unobserved ablation.
The dynamic model fixes density at 3000 kg/m3, free-molecular drag Gamma=1
(equivalent to Cd=2 with a one-half drag convention), and sigma=1e-8 s2/m2.
Dynamic-mass bounds are projections of a whole-pulse multiplier-calibrated
joint likelihood region, not Bayesian mass quantiles or bootstrap percentiles.
RCS mass intervals propagate event-wide RMS scatter in dBsm through the lookup
and time integration; scattering-model uncertainty is not included.
Apparent-path residuals are chirp-coupled diagnostics, not independent 3-D
position measurements. Detector-derived apparent track rates are not
independent Doppler velocity measurements.

CODE AND REPRODUCTION
source/lfm_meteor/ is the current processing snapshot. Its tracked working-tree
changes are explicitly recorded in source/provenance.h5; the source files,
not only a commit label, are archived. source/calibrated-fit-run/ is the exact
2328909207f68982b8aed4b19fb3046f61fe933c commit used for the mass-confidence run.
source/fdtd/ and source/rustplotlib/ retain the previously archived solver and
figure-library snapshots. Existing licenses are preserved; no blanket license
is inferred for code without a declaration. Python/Cargo/system dependencies
are not vendored. Native fits require Cargo, system HDF5, and MPI as documented
in the source. Full catalogue scripts may need local path relocation.

Verify hashes, selections, mass intervals, and the reported influx ranges:
  conda run -n base python source/reproduction/verify_current_release.py --root .
Dependencies: numpy and h5py. This is a data/product consistency check, not
a rerun of the full raw-voltage fitting or an independent physical validation.
To use the archived population scripts in their original directory layout,
place or link the extracted results/ directory at source/lfm_meteor/results/.

No access credentials or private reviewer tokens are included in these files.
