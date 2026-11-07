Sanya Tri-static Meteor Head Echo Experiment
Supporting data and analysis code
Reserved DOI: 10.5281/zenodo.24037714

This record is an unpublished reviewer draft. Public release is planned upon
acceptance of the paper. Reviewer access is supplied separately.

DOWNLOADS
1. sanya-raw-data.tar.gz: 5,215 event-only voltage cuts from Sanya, Danzhou
   and Wenchang, with detected times, ranges, signal strengths and selections.
2. sanya-results.tar.gz: fitted trajectories, full saved uncertainty sources,
   heliocentric orbits, RCS lookup, dynamic masses and observed RCS mass losses,
   radiant exposures, detection yields and mass-influx calculations.
3. sanya-code.tar.gz: frozen processing source, Rust physical-waveform solver,
   Rustplotlib, FDTD source, and a tested single-event replication driver.
4. README.txt: this guide.
5. checksums.h5: SHA256/MD5 checksums and byte sizes of downloads 1--4.

Extract all three archives into the same empty directory. The folders are:
  L0/                    Original event-only raw-voltage cuts by station.
  L1/                    Detected quantities, event fits and exclusions.
  uncertainty/           649 current uncertainty-source files for 232 events.
  orbits/                Propagated orbit ensembles and initial-state snapshots.
  L1_velocity/           Apparent track rates and inferred state vectors.
  lookup/                RCS lookup inputs and limitations (see archive paths).
  mass-derived/          154 dynamic masses, 195 RCS masses, 133 paired events,
                         size statistics, influx inputs and resampling draws.
  kinematics/            Geometry, radiants, exposure, beam area and calibration.
  yield/                 Observing-duration and survey-yield products.
  selection-inputs/      Unique selection/audit inputs not already in event fits.
  replication/           Results of the single-event replication test.
  source/                Frozen source repositories and replication driver.
Internal evidence copies that were byte-identical to these products have been
removed. Scientific data are unchanged. Member-level hashes and source mapping
are in source/reproduction/inventory.h5 within the code archive.

SINGLE-EVENT REPLICATION
Event tri_0012_1713798481419350028 has 48 fitted station-pulse segments:
22 Sanya, 15 Danzhou and 11 Wenchang.

From the common extraction directory, build the archived solver:
  cd source/lfm_meteor/tristatic_native
  cargo build --release --locked
  cd ../../..
Then use a Python environment with numpy, scipy and h5py:
  python source/reproduction/replicate_event.py --root .
  python source/reproduction/replicate_event.py --root . --raw-cuts
For the author's base conda environment, prefix Python commands with
  conda run -n base
The default native library suffix is selected for macOS/Linux. An alternative
library can be passed with --library /absolute/path/to/library.

The first run uses the complete archived 1,280-sample fitting windows.
The second reconstructs 1,000-sample windows directly from the three L0 cuts.
The L0 event cuts omit outer noise padding preserved in L1/input.h5.
All 52,729 overlapping raw samples matched exactly in the verification run.
Re-optimization starts from the archived pre-fit guess, not the saved optimum.
The raw-cut run reproduced the trajectory within 0.104 m in ECEF position and
0.51 m/s in speed; the initial radius was 41.725 versus 41.696 micrometres.
It reuses the archived frame transforms, atmosphere, noise model, calibration
and initial guess. It does not rerun posterior/bootstrap sampling, orbit
propagation or population/influx calculations. Acceptance checks are stored
in the replication HDF5 outputs.

RUNTIME
Python/Cargo registry dependencies and system HDF5/MPI/TeX are not vendored.
Observed Python distributions are in source/environment.h5. Full catalogue
drivers can require further dependencies (including arviz_stats), absolute-path
relocation, and Apple Metal tools for some FDTD calculations. The standalone
replication driver avoids catalogue and posterior-only Python dependencies.
On the tested Mac, rustc 1.99.1 produced a loadable native library; the local
Homebrew rustc 1.97.1 build was rejected by macOS for LINKEDIT alignment.
No archived Rust source change was needed for the test.

DATA INTERPRETATION
232 accepted trajectories/orbit ensembles are retained. The 235 fitted-event
folders also preserve one rejected and two pending-refit events for traceability;
use the accepted-event identifiers in the catalogue/selection products.
Dynamic mass is the modeled mass remaining at first detection. RCS mass is
the inferred mass lost during the observed in-beam interval. Neither includes
unobserved mass loss before detection. The selected uncertainty sources include
conditional posteriors, time-block bootstraps and one random-deletion robustness
ensemble; their interpretation is recorded in each product.
The RCS lookup retains its low-speed override and boundary-clamping metadata;
mesh/time convergence is not certified. Influx velocity/altitude corrections
are heuristic. Apparent path residuals are chirp-coupled path-coordinate
residuals, not independent measurements of absolute 3-D position accuracy.
No continuous raw recordings, private reviewer tokens or access credentials
are included.

LICENSES
Data: CC BY 4.0. Existing code declarations are preserved; Rustplotlib is MIT.
No blanket lfm_meteor/FDTD project-wide license has been inferred.
