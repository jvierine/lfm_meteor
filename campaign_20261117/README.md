# November 2026 Tri-static Memo

This campaign is separate from the April paper catalogue. It reads station
products under `/gpfs/pub/data_other/juha/20261117_<station>` and writes results
under `/gpfs/pub/data_other/juha/20261117_tristatic`.

Submit from `/home/juha/src/lfm_meteor`:

```bash
sbatch campaign_20261117/run.slurm
```

The job requests 128 CPUs on node03. Numerical work runs in the Slurm allocation,
not on the head node. The first run should be bounded with `--limit 4` to check
geometry and raw-voltage conventions before launching the full catalogue.

Associations are unique time-overlap candidates, not confirmed trajectories.
The first fit uses the recorded gate delays and the existing Rust voltage
kernel, without whitening. Do not substitute the April hardware correction.
Hardware timing, physical mass estimates, absolute RCS calibration, and orbit
uncertainties require campaign-specific checks before results are merged into
the article. Failures remain visible in the HDF5 fit-status table and event logs.
