the data is on j@revontuli.uit.no. that is the "server" for this project

Data is in (revontuli.uit.no):
/mnt/data/juha/SANYA/Juha/20240422
Code is in (revontuli.uit.no):
/home/j/src/lfm_meteor

Python/compute environment on revontuli:
- The noninteractive SSH shell does not expose `conda`; `conda run -n base ...`
  and `python` are not available there by default.
- Use `/usr/bin/python3` for this project on revontuli.  It has the required
  packages verified for the trajectory fitting workflow: `numpy`, `h5py`,
  `scipy`, `astropy`, `mpi4py`, `pymsis`, and `jcoord`.
- Use `/usr/bin/mpirun` for MPI jobs.  Example:
  `cd /home/j/src/lfm_meteor_runs/<run_dir> && mpirun -np 48 python3 <script>.py`
- Locally on this Mac, still follow the global `~/agents.md` preference and use
  `conda run -n base python ...` for Python commands.


Edit code locally on this computer, push changes to git, and 

The github repo is:
git@github.com:jvierine/lfm_meteor.git

on this computer it is in ~/src/lfm_meteor

The slide deck: 
/Users/jvi019/Dropbox/Work/Documents/2026/cas_visit/sanya_head_echoes.key

documents progress

Always version the data output with processing script version.

Project pairing:
- This is the code/data repository for the Sanya tri-static meteor project.
- Pair this repository with the article repository at `/Users/jvi019/src/sanya_tristatic_paper`.
- Use `/Users/jvi019/src/lfm_meteor` for processing code, data products, and figure-generation scripts.
- Use `/Users/jvi019/src/sanya_tristatic_paper` for article text, tables, figures copied into the paper, and paper memos.

Trajectory fitting delay convention:
- Memo 3 in `/Users/jvi019/src/sanya_tristatic_paper` is the controlling
  reference for station delay handling.
- Use one common zero-gate tx-target-rx delay for Sanya, Danzhou, and
  Wenchang: 359.45540369317763 us.
- This comes from the Sanya satellite-calibrated raw first-sample delay
  466.320 us and the -16.0186 km Sanya range correction.
- The corresponding common zero-gate tx-target-rx path length is
  107.76201901456 km.
- Use the same equation for every path:
  L_i(g_i) = c * (359.45540369317763 us + g_i / f_s).
- Do not use a Sanya-only monostatic `2 * range` observable in the ballistic
  fit. The fitted Sanya, Danzhou, and Wenchang observables are all
  tx-target-rx path lengths.
