Native Sanya article figures
===========================

Python is a valid Rustplotlib interface. Matplotlib is not: it must not provide
layouts, paths, transforms, text coordinates, colourbars or scene files.
The rejected Matplotlib RendererBase converter has been removed.

Scientific HDF5 arrays are the input boundary for the current Rust builders.
Python may compute scientific quantities and call Rustplotlib's own API, but
must not construct a Matplotlib figure for Rust to trace. The native binary
explicitly rejects the former graphics-scene schema.

Migration status:

* Figure 1: native perspective geometry in src/native/geometry.rs; the original
  WebGL image remains available as an archival reference.
* Figure 10: native builder in src/native/beat.rs, reading original complex
  voltage, time, FFT-frequency and power arrays directly from scientific HDF5.
* Figure 8: native builder in src/native/radiants.rs; Rustplotlib computes the
  Hammer projection, native contours, graticules, frame, symbols and annotations.
  prepare_native_radiant_data.py only calculates scientific visibility hours
  with Astropy/NumPy and writes HDF5; it never imports Matplotlib.
* Figure 2: src/native/detection.rs, with native logarithmic counts, solar-time
  secondary axis, scientific noise data and receiver-temperature reference lines.
* Figure 3: src/native/height.rs, with a shared-height display and a square beam
  histogram panel, native antenna contours and no meteor-path overlays.
* Figures 4--5: src/native/event.rs, preserving measurement masks, asymmetric
  radius-profile mass intervals and forward-model uncertainty bands. Preparation
  evaluates the existing scientific forward models; it never calls plot code.
* Figure 6: src/native/mass.rs, same 110-event selection and asymmetric mass
  bounds; mass histogram and speed scatter share the figure, with diameter above.
* Figure 7: src/native/rcs.rs, preserving pulse/event samples and saved intervals.
* Figure 9: src/native/orbits.rs, native percentiles and classification from the
  100-sample orbit arrays; same five visual-only exclusions, 229 events retained.

All ten manuscript figures have native Rust builders. There is no converter
fallback. Every builder uses an A&A ArticleLayout with 184 mm available width
and 10 TeX-point text. Exports trim blank outer margins without scaling the
content. Include the generated .include.tex companions directly: expanding a
cropped PDF to linewidth would change its typography. The companions check the
article linewidth and body font before including at the physical export width.
Validation checks actual PDF dimensions, 10-point printed text and outer ink
margins below 1.7 mm. It does not compile the article.

Build and validate::

    conda run -n base python prepare_native_paper_data.py height
    conda run -n base python prepare_native_paper_data.py event --event tri_0162_1713823274259349823
    conda run -n base python prepare_native_paper_data.py event --event tri_0022_1713803698934349060
    conda run -n base python prepare_native_paper_data.py rcs
    conda run -n base python prepare_native_paper_data.py detection
    conda run -n base python prepare_native_radiant_data.py
    conda run -n base python build_paper_figures_rustplotlib.py
    conda run -n base python validate_paper_rustplotlib.py

The launcher only invokes Cargo/the Rust binary. Direct execution without
Python is also supported::

    HDF5_DIR=/opt/anaconda3 cargo build --release --bin native-paper-figures
    DYLD_LIBRARY_PATH=/opt/anaconda3/lib target/release/native-paper-figures beat ../results/appendix_dechirped_beat_example_v20260702b.h5 ../results/paper_rustplotlib_native/appendix_dechirped_beat_example

Native outputs are separate in results/paper_rustplotlib_native. After PDF
review, --install-only copies validated PDF/PNG/SVG outputs into the paper.
These commands do not compile the article.

The scientific preparer blocks Matplotlib imports at runtime. Legacy scientific
modules were adjusted to import plotting dependencies only inside their old
plot functions, which are never called by this pipeline. Noise preparation uses
the same GSM2008 PCA/PCHIP calculation and RING pixel indexing as PyGDSM, via
SciPy and Astropy HEALPix, avoiding its incidental plotting imports. Fits are not
rerun; raw measurements and retained masks remain unchanged.

Alternative Figure 1 geometry
-----------------------------

``src/native/geometry.rs`` uses Rustplotlib's native ``PerspectiveCamera`` and
vector paths, transparent beam triangles, physical-size scatter markers and
LaTeX labels. ``prepare_native_geometry_data.py`` copies the original WebGL
file's scientific ENU coordinates and beam meshes into HDF5; it does not import
the original plotting program or a graphics backend. The camera is reconstructed
from station/common-volume reference positions in the supplied image, not an
exact saved interactive camera state. Beam widths retain the original 2x visual
expansion. The original Figure 1 and its manuscript reference remain unchanged.

Build, validate and install the separate PDF/PNG/SVG version::

    conda run -n base python build_geometry_rustplotlib.py

Outputs: ``figures/rustplotlib/sanya_system_geometry.*`` in the paper repository,
including the checked ``.include.tex`` companion used by the manuscript.
Source and output hashes are in
``results/paper_rustplotlib_native/sanya_system_geometry_validation.h5``.
