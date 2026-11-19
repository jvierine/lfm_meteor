# Incremental Event Publication

The initial implementation deferred event cuts/plots until every raw file was
indexed. The user explicitly rejected that workflow. It is replaced with the
native pipeline::Live coordinator, not a separate plotting/reduction script.

MPI rank zero loads completed per-file indexes in chronological manifest order,
retains a bounded neighbouring-file cache, and publishes raw HDF5 cuts plus
PNG immediately when enough future coverage exists for the maximum allowed
event length, inter-detection gap and time padding. Other MPI ranks process
raw files. Serial mode polls the same coordinator after each file. Ownership
is the source file of the first RANSAC inlier, preventing duplicate boundary
events. End-of-dataset cuts explicitly flag clipped padding. Noise rows are
collected once as each index is loaded; final noise/catalogue are written after
the scan, without re-reading every index.

Internal indexes now live in work/; legacy root indexes are readable for resume.
The external run's completed indexes are moved to work without recomputation.
The first three verified pilot cuts/plots were placed on the external disk while
the workflow was corrected; the live coordinator replaces them using full-run
metadata and continues numbering chronologically.

A regression processes two out of three synthetic input files and asserts that
event_000000.h5 already exists while the third index and final catalogue do not.
The existing cross-file, weak-pulse, raw-byte, timing and I/O tests also pass.
The overlong counter is named overlong_window_rejections, since neighbouring
windows can count the same rejected long sequence; it is not a meteor count.
