# Tri-static voltage review

The HTTP service runs on Avaruus loopback port 8767. The browser uses the
rustplotlib Rust/WASM `SeriesView` axes and `LinePlot` WebGL2 data layer.
Generated WASM assets are separate from source code. Transfer source through git;
transfer only generated HDF5 data and compiled WASM assets with rsync.

Current deployment:

- Source: `/home/juha/src/lfm_meteor_review_20261029/voltage_review/`
- Data: `/home/juha/src/lfm_meteor/results/tristatic_voltage_correlated_20261029/`
- Reviews: `/home/juha/src/lfm_meteor/results/voltage_review_flags/`
- Token: `/home/juha/src/lfm_meteor/results/voltage_review.token` (private)
- Log: `/home/juha/src/lfm_meteor/results/voltage_review.log`

Access is via the Mac's SSH tunnel at `http://127.0.0.1:8767/` with a bearer
token supplied in the URL fragment or login field. The fragment is removed
from the address bar and the token is retained only in session storage.
The launch agent `com.jvi019.sanya-review-tunnel` reconnects the tunnel.

Click a measurement to toggle its selection, then use Flag selected or Restore
selected. Selection is linked across panels. Plain drag pans; Ctrl/Cmd-wheel
zooms; Reset view or right-click resets. C clears selection; B toggles event bad.
Undo creates a new saved revision restoring the previous review state.

Flags use station and exact UTC Unix nanoseconds, never display coordinates or
rounded floating-point timestamps. Each HDF5 review file stores the complete
measurement identity array, manual rejection mask, version, note, bad-event
state, source-data fingerprint, and revision history. Writes are atomic and
reject stale concurrent edits. Fit products are not modified. Flagging does not
refit: the plotted curve and beat check still refer to the original fit.

`export_voltage_plot.py` creates compact `plot.h5` bundles from completed fits.
Refresh reads new bundles already transferred to Avaruus; it does not transfer
new Revontuli results. Failed events without fit products have no plot bundle.

Tests: `conda run -n base python -m pytest voltage_review/test_server.py -q`.
`test_browser.cjs` is for a disposable flags directory; it deliberately writes
three test revisions and must not be run against the production review store.
