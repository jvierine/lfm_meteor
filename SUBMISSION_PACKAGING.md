# Sanya Submission Packaging

Status: **UNFROZEN INITIAL CORE**. No upload or publication is implemented.
The parent/user performs any reviewer-only Zenodo draft upload in Chrome.
Do not publish. Do not transfer continuous raw recordings.

## Scope and ownership

This task owns only `package_submission.py`, its tests, and this document.
Do not edit manuscript, mass/orbit scripts, original measurements, or fit outputs.
All HDF5 inputs are copied unchanged. The L1 detection file copies existing
groups without estimating or relabelling velocity. Source snapshots contain
allowlisted committed Python/Rust source, Cargo manifests/locks and this guide,
not a repository history, home directory, credentials, binaries, or caches.
No CSV output is written. Historical CSV validation tables are not packaged.

## Remote staging

Use a fresh destination for every attempt. Code transport is Git only, into an
isolated checkout; never rsync source. Local Python uses `conda run -n base`.
On avaruus the verified interpreter is `/home/juha/miniforge3/bin/python`;
on revontuli the project instructions specify `/usr/bin/python3`.

```sh
# avaruus, from the isolated packaging checkout
/home/juha/miniforge3/bin/python package_submission.py \
  --component catalog --root /home/juha/src/lfm_meteor \
  --output /home/juha/sanya_submission_staging/20261107/catalog-core

# revontuli, from the isolated packaging checkout
/usr/bin/python3 package_submission.py --component cuts \
  --root /mnt/data/juha/sanya/replication_data \
  --output /mnt/data/juha/sanya/submission_staging/20261107/cuts-core
```

Each component has `manifest.h5` (file sizes, SHA256, provenance and pending
requirements) and `checksums.h5` (archive and manifest SHA256 and byte sizes).
Archives are separated into raw cuts, detections, fits, lookup and code.
Files have stable archive order, owner, modes and timestamps. Each archive is
read back and hashed member by member. Source signatures are checked before
and after staging; changed inputs fail the build. This detects ordinary changes,
but is not a substitute for a coordinated immutable scientific freeze.
An interrupted build has no verified completion; retain it for diagnosis and
retry with a new destination. No input or existing output is overwritten.

Use rsync only for the generated archives/manifests. After transfer, verify
archive SHA256 against `checksums.h5`, before any manual browser upload.

## Contents and interpretation

- L0: memo16 station detection cuts only, with receive voltage, pulse timestamps,
  science window, matched-filter support and detection reference groups.
- L1 detections: original range, UTC time, SNR, cut selection and station IDs.
  The original cuts do not supply measured Doppler velocity. Do not substitute
  fitted scalar speed or a numerical range derivative and call it measured.
- Fits: per-event input, fit, physical positions/speeds, plot intervals, beam
  RCS/SNR, initial-radius/initial-mass posterior, observed RCS mass-loss posterior,
  exclusions and compact validation products. Original unexcluded voltage inputs
  are retained when available. Likelihood caches and full trajectory chains are
  excluded from this compact initial core; their final inclusion needs an owner
  decision, especially for reproducing posterior uncertainties.
- RCS lookup: exact standalone table named in each mass product, including its
  low-speed override metadata. Preserve the collisionless/head-on pilot warning
  and lack of certified mesh/time convergence.
- `m_RCS` means observed ablated mass over bounded in-beam intervals, not initial
  meteoroid mass. Its scatter interval is conditional, not total uncertainty.
- Existing source paths remain only in HDF5 metadata/code. Portable path mapping
  and historical producer revisions still need verification. The committed source
  baseline is not asserted to be the exact producer of all historical products.
- Rustplotlib is an external Cargo path dependency and is not yet vendored.

## Freeze coordination

Keep these archives marked unfrozen even if uploaded to a private draft.
The coordinating parent must obtain final-product paths and producer commits
from the mass, orbit, selection/exclusion, and uncertainty owners, plus explicit
confirmation that their writers have stopped. No owner acknowledgements have
been received by this packaging task. The current 20261029 directory is a
starting point, not an implicit approval of final science products.

Before final freeze, rebuild into a NEW versioned staging directory and check:

1. Catalogue-to-L0 station event joins, including UTC versus legacy UTC+8 IDs.
2. All required measured range/time/Doppler velocity/SNR fields and conventions.
3. Individual position AND velocity states, uncertainty/covariance, r0/m0,
   m_RCS, final exclusions and their revisions; do not invent absent products.
4. Each beam/mass product's input hashes versus the packaged fit/flags/lookup.
5. Final mass/orbit products and their source code, external dependencies,
   licenses, calibrations, atmosphere inputs, and portable path resolution.
6. Rerun memo16 matched-filter validation into HDF5 (the historical validation
   is CSV and is intentionally not copied into the new data product).
7. Verify downloaded archive hashes and inspect Zenodo draft privacy in Chrome.

Only the parent/user handles the Zenodo UI. A local ready-to-select file is not
an upload, a publication, or evidence that scientific freeze gates passed.

## 2026-11-07 Handoff

The user confirmed reserved DOI `10.5281/zenodo.24037714`, unpublished draft
24037714. All nine authors are listed in the local handoff README in manuscript
order. Data CC BY 4.0 is the requested draft default; there is no project-wide
code LICENSE declaration in the inspected repository, so do not invent one.

The corrected orbit component includes 232 ensembles and 229 display-selected
events, excludes tri0035, and replaces tri0145. Its 232 saved state snapshots
and individual propagated results are included, not the old aggregate orbit
catalogue. All saved samples are bound; this is not a physical certainty claim.
The orbit-source snapshot includes solver-convention commit `1ee5028` and the
two assembly/verification scripts. Other agents' source was captured without
editing it, on an isolated Git branch, not committed to their active branch.

Remote provenance audit against the saved propagation hashes found all 232
fits, all 232 inputs, and all 649 full uncertainty sources matching on avaruus.
Thus the 31 local input mismatches and 647 locally absent source files are
local-copy limitations, not missing remote originals. The audit HDF5 retains
each expected and actual hash and path. The uncertainty component packages
these verified originals separately; it does not rerun inference.

For transferred archives without unpacked staging trees:

```sh
conda run -n base python package_submission.py --verify --archives-only \
  --output results/submission_staging/20261107/cuts-core
```

Use `orbits-current/code.tar.gz` in preference to the older code snapshots
inside the initial cuts/catalog component directories. Component manifests
remain immutable records of those initial snapshots.

The initial handoff audit found 5,215 station cuts (4,546 Sanya, 368 Danzhou,
301 Wenchang). All 705 station-cut associations for the 235 fitted events are
present. All 235 beam-to-fit, RCS-to-fit and RCS-to-beam recorded hashes match.
All 235 fits supply initial radius and mass intervals; all 235 per-detection
physical-state files still lack velocity vectors. This is a pending export,
not a reason to relabel scalar speed as velocity. The 235-event fit archive
retains rejected/pending events for audit; final scientific orbit exclusions
are explicitly supplied by the corrected 232-event orbit component.

`audit_submission_handoff.py` reproduces the upload-file size/hash inventory
and these checks using only the generated local archives and HDF5 metadata.
It writes a new HDF5 file, never CSV, and leaves all archives unchanged.

The raw-cut archive was read back remotely and locally, including every
member checksum. Its immutable identity is 1,009,707,600 bytes and SHA256
`915e3750eadad21989053deedb838ed51393b1d3b809a8020fe4517f923d7e83`.
Raw cuts may be uploaded independently while final parsed products/code await
the coordinating parent's freeze approval. Do not replace the final code
snapshot until the parent supplies the remaining producer commits.

## L1 Completion and Current Uncertainty Audit

`audit_submission_uncertainty.py` checks CURRENT remote plot-selected methods,
not simply whether old files still exist. All 235 archived plots and dynamic
mass products match avaruus, and their methods/quantiles agree. The accepted
232 select 139 conditional posteriors, 92 block bootstraps and one random-deletion
ensemble, using the same 649 archived source files. The temporary claim that
40 orbit ensembles were superseded was retracted: 40 LOCAL plots were stale,
but saved orbit snapshots already used the correct remote bootstrap sources.
The new `orbits-refreshed` package contains the corrected audit and full replay.

`export_submission_l1.py` reads only verified staged archives and adds:

- `detection_rates.h5`: 68,979 apparent half-path/total-path range-rate estimates
  for 5,215 station events, using the existing quadratic detection-track
  convention. These are measurement-derived rates, not independent Doppler
  velocities or three-dimensional object speeds. LFM delay/Doppler coupling
  remains. OLS standard errors are conditional on independent residuals.
- `inferred_states.h5`: nominal-time best-fit GCRS/ECEF positions/velocities
  for all 14,859 points in 235 events, reconstructed with the stored frame maps
  and fixed GCRS direction. No trajectory is refitted. All velocities and SNRs
  are finite. All 908 excluded points remain present with `used_in_fit=false`.
  Final event exclusions 0020/0096/0035 and accepted232 are explicit.
- Sanya calibrated RCS availability, r0/m0, observed RCS mass loss, published
  scalar intervals and current uncertainty-method labels. Non-Sanya calibrated
  RCS and independent detector Doppler are explicitly unavailable, not silently
  substituted. Vector covariance is not claimed; full selected parameter
  ensembles remain in the unchanged uncertainty archive.

Unit tests cover exact quadratic rates and frame-velocity transformation.
Stored position/fit-direction consistency is within 2.3e-9 m numerically.
Finite-difference velocity errors have a large tail during strongly decelerating
excluded extrapolations, largest for rejected0035; finite differences are NOT
used to define the inferred velocity vectors. These checks verify export
consistency, not calibration or physical validity of rejected trajectories.

Use `l1-complete-archive`, not the earlier development `l1-velocity-archive`.
`yield-current` includes the parent-produced yield HDF5 only after its stored
selection hash matches the refreshed orbit product. The final code archive
must include the parent yield script and the fresh orbit scripts after the
coordinating parent's commit. No reviewer capability URL/token belongs in
these files or in Git; only the confirmed reserved DOI is recorded.
