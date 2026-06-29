#!/usr/bin/env python3
"""Browse joint delay/Doppler event plots and relaunch individual refits."""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Button, LassoSelector

import fit_gcrs_trajectories_lfm_ambiguity as gfit
import fit_event_joint_delay_doppler_fft as joint_fit


DEFAULT_RESULTS_DIR = Path("results/tristatic_calibrated_chirp_v20260624b")
DEFAULT_SNR_MIN_DB = 15.0
SITE_LABELS = ("Sanya", "Danzhou", "Wenchang")
SITE_COLORS = ("#4c78a8", "#f58518", "#54a24b")


def reserve_keybindings():
    reserved = {"left", "right", "r", "R", "f", "F", "g", "G", "c", "C"}
    for key in list(plt.rcParams):
        if key.startswith("keymap."):
            plt.rcParams[key] = [value for value in plt.rcParams[key] if value not in reserved]


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def event_id_from_path(path: Path) -> str:
    stem = path.stem
    prefix = "joint_delay_doppler_fft_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def discover_events(results_dir: Path):
    paths = sorted(Path(p) for p in glob.glob(str(results_dir / "joint_delay_doppler_fft_tri_*.png")))
    rows = []
    for png in paths:
        event_id = event_id_from_path(png)
        base = png.with_suffix("")
        rows.append({"event_id": event_id, "base": base, "png": png, "h5": base.with_suffix(".h5")})
    return rows


def h5_summary(path: Path):
    if not path.exists():
        return {"status": "missing h5"}
    out = {}
    try:
        with h5py.File(path, "r") as h:
            out["event_id"] = decode_attr(h.attrs.get("event_id", path.stem))
            if "joint_fit" not in h:
                out["status"] = "missing joint_fit"
                return out
            g = h["joint_fit"]
            for key in (
                "dynamical_model",
                "fallback_reason",
                "bad_fit_detected",
                "bad_fit_reasons",
                "bad_fit_recovery_step",
                "pre_recovery_bad_fit_reasons",
                "n_points",
                "n_path_observations",
                "n_fft_observations",
                "n_delay_clipped_observations",
                "n_coincident_delay_constraint_rows",
                "coincident_delay_weight",
                "rms_total_path_residual_m",
                "mean_abs_total_path_residual_m",
                "rms_fft_residual_hz",
                "mean_abs_fft_residual_hz",
                "rms_path_rate_residual_mps",
                "mean_abs_path_rate_residual_mps",
                "fit_mode",
                "fit_epoch_time_ns",
            ):
                if key in g.attrs:
                    value = g.attrs[key]
                    out[key] = decode_attr(value) if isinstance(value, (bytes, str)) else value
            out["status"] = "ok"
    except Exception as exc:
        out["status"] = f"h5 read error: {exc}"
    return out


def truthy(value) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def beat_residual_khz_to_total_path_rate_mps(freq_khz):
    return -gfit.RADAR_WAVELENGTH_M * np.asarray(freq_khz, dtype=np.float64) * 1e3


def total_path_rate_mps_to_beat_residual_khz(path_rate_mps):
    return -np.asarray(path_rate_mps, dtype=np.float64) / gfit.RADAR_WAVELENGTH_M / 1e3


def default_manual_outlier_h5(results_dir: Path) -> Path:
    return results_dir / "manual_outliers.h5"


def read_event_arrays(path: Path):
    if not path.exists():
        return None
    with h5py.File(path, "r") as h:
        if "joint_fit" not in h:
            return None
        g = h["joint_fit"]
        out = {
            "time_ns": np.asarray(g["time_ns"][:], dtype=np.int64),
            "t_rel_s": np.asarray(g["t_rel_s"][:], dtype=np.float64),
            "path_residuals_m": np.asarray(g["path_residuals_m"][:], dtype=np.float64),
            "path_keep": np.asarray(g["path_keep"][:], dtype=bool),
            "fft_residuals_hz": np.asarray(g["fft_residuals_hz"][:], dtype=np.float64),
            "fft_keep": np.asarray(g["fft_keep"][:], dtype=bool),
            "path_rate_residuals_mps": np.asarray(g["path_rate_residuals_mps"][:], dtype=np.float64),
        }
    return out


def load_manual_masks(path: Path, event_id: str, time_ns: np.ndarray):
    n = len(time_ns)
    delay = np.zeros((n, 3), dtype=bool)
    fft = np.zeros((n, 3), dtype=bool)
    if not path.exists():
        return delay, fft
    with h5py.File(path, "a") as h:
        if event_id not in h:
            return delay, fft
        g = h[event_id]
        stored_time = np.asarray(g.get("time_ns", []), dtype=np.int64)
        if stored_time.shape != time_ns.shape or not np.array_equal(stored_time, time_ns):
            return delay, fft
        if "delay_outlier" in g:
            value = np.asarray(g["delay_outlier"][:], dtype=bool)
            if value.shape == delay.shape:
                delay = value
        if "fft_outlier" in g:
            value = np.asarray(g["fft_outlier"][:], dtype=bool)
            if value.shape == fft.shape:
                fft = value
    return delay, fft


def save_manual_masks(path: Path, event_id: str, time_ns: np.ndarray, delay: np.ndarray, fft: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as h:
        if event_id in h:
            del h[event_id]
        g = h.create_group(event_id)
        g["time_ns"] = np.asarray(time_ns, dtype=np.int64)
        g["delay_outlier"] = np.asarray(delay, dtype=bool)
        g["fft_outlier"] = np.asarray(fft, dtype=bool)


class EventPlotBrowser:
    def __init__(
        self,
        results_dir: Path,
        start_event_id: str | None,
        snr_min_db: float,
        extra_fit_args: list[str],
        manual_outlier_h5: Path | None = None,
        random_initial_guesses: int = 24,
        coincident_delay_weight: float = joint_fit.DEFAULT_COINCIDENT_DELAY_WEIGHT,
    ):
        self.results_dir = results_dir
        self.snr_min_db = float(snr_min_db)
        self.extra_fit_args = list(extra_fit_args)
        self.manual_outlier_h5 = manual_outlier_h5 or default_manual_outlier_h5(results_dir)
        self.random_initial_guesses = int(random_initial_guesses)
        self.coincident_delay_weight = float(coincident_delay_weight)
        self.rows = discover_events(results_dir)
        if not self.rows:
            raise RuntimeError(f"No event PNGs found in {results_dir}")
        self.index = 0
        if start_event_id:
            matches = [idx for idx, row in enumerate(self.rows) if row["event_id"] == start_event_id]
            if not matches:
                raise ValueError(f"Event {start_event_id!r} not found in {results_dir}")
            self.index = matches[0]
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.message = ""
        self.event_arrays = None
        self.delay_outlier = None
        self.fft_outlier = None
        self.pick_map = {}
        self.fft_secondary_axis = None
        self.lasso = None
        self.lasso_fft = None
        self.lasso_axes_kind = {}

        self.fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))
        plt.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.90, hspace=0.32, wspace=0.26)
        self.ax_top_left = axes[0, 0]
        self.ax_top_right = axes[0, 1]
        self.ax_delay = axes[1, 0]
        self.ax_fft = axes[1, 1]
        self.status_text = self.fig.text(0.07, 0.055, "", ha="left", va="bottom", fontsize=8.2, family="monospace")
        self.msg_text = self.fig.text(0.07, 0.025, "", ha="left", va="bottom", fontsize=9)
        self.buttons = {
            "prev": Button(self.fig.add_axes([0.55, 0.035, 0.07, 0.045]), "Left"),
            "next": Button(self.fig.add_axes([0.63, 0.035, 0.07, 0.045]), "Right"),
            "delay_fit": Button(self.fig.add_axes([0.72, 0.035, 0.10, 0.045]), "Delay (F)"),
            "joint_fit": Button(self.fig.add_axes([0.84, 0.035, 0.11, 0.045]), "Joint (G)"),
        }
        self.buttons["prev"].on_clicked(lambda _event: self.goto(self.index - 1))
        self.buttons["next"].on_clicked(lambda _event: self.goto(self.index + 1))
        self.buttons["delay_fit"].on_clicked(lambda _event: self.refit_current("delay-only"))
        self.buttons["joint_fit"].on_clicked(lambda _event: self.refit_current("joint"))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.timer = self.fig.canvas.new_timer(interval=500)
        self.timer.add_callback(self.poll_refit)
        self.timer.start()
        self.setup_lasso()
        self.load_current()

    def setup_lasso(self):
        if self.lasso is not None:
            try:
                self.lasso.disconnect_events()
            except Exception:
                pass
        if self.lasso_fft is not None:
            try:
                self.lasso_fft.disconnect_events()
            except Exception:
                pass
        self.lasso_axes_kind = {self.ax_delay: "delay", self.ax_fft: "fft"}
        self.lasso = LassoSelector(
            self.ax_delay,
            onselect=lambda verts: self.on_lasso("delay", verts),
            button=1,
        )
        self.lasso_fft = LassoSelector(
            self.ax_fft,
            onselect=lambda verts: self.on_lasso("fft", verts),
            button=1,
        )

    def current(self):
        return self.rows[self.index]

    def load_current(self):
        row = self.current()
        for ax in (self.ax_top_left, self.ax_top_right):
            ax.clear()
            ax.axis("off")
        try:
            image = mpimg.imread(row["png"])
            self.draw_top_image_panels(image)
        except Exception as exc:
            self.ax_top_left.text(0.5, 0.5, f"Could not read {row['png']}:\n{exc}", ha="center", va="center")
        self.event_arrays = read_event_arrays(row["h5"])
        if self.event_arrays is not None:
            self.delay_outlier, self.fft_outlier = load_manual_masks(
                self.manual_outlier_h5,
                row["event_id"],
                self.event_arrays["time_ns"],
            )
        else:
            self.delay_outlier = np.zeros((0, 3), dtype=bool)
            self.fft_outlier = np.zeros((0, 3), dtype=bool)
        self.draw_residual_axes()
        summary = h5_summary(row["h5"])
        bad = truthy(summary.get("bad_fit_detected", False))
        title_color = "crimson" if bad else "black"
        self.fig.suptitle(
            f"{self.index + 1}/{len(self.rows)}  {row['event_id']}",
            color=title_color,
            fontsize=12,
            fontweight="bold" if bad else "normal",
        )
        self.status_text.set_text(self.status_string(row, summary))
        self.msg_text.set_text(self.message)
        self.fig.canvas.draw_idle()

    def draw_top_image_panels(self, image):
        h, w = image.shape[:2]
        top = image[: h // 2, :, :]
        left = top[:, : w // 2, :]
        right = top[:, w // 2 :, :]
        self.ax_top_left.imshow(left)
        self.ax_top_right.imshow(right)

    def draw_residual_axes(self):
        self.pick_map = {}
        for ax in (self.ax_delay, self.ax_fft):
            ax.clear()
            ax.axhline(0.0, color="0.25", lw=0.9)
            ax.grid(True, alpha=0.25)
        if hasattr(self, "fft_secondary_axis") and self.fft_secondary_axis is not None:
            try:
                self.fft_secondary_axis.remove()
            except Exception:
                pass
            self.fft_secondary_axis = None
        if self.event_arrays is None:
            self.ax_delay.text(0.5, 0.5, "No HDF5 data", ha="center", va="center", transform=self.ax_delay.transAxes)
            self.ax_fft.text(0.5, 0.5, "No HDF5 data", ha="center", va="center", transform=self.ax_fft.transAxes)
            return
        t = self.event_arrays["t_rel_s"]
        for col, (label, color) in enumerate(zip(SITE_LABELS, SITE_COLORS)):
            self.plot_click_points(
                self.ax_delay,
                "delay",
                col,
                t,
                self.event_arrays["path_residuals_m"][:, col],
                self.event_arrays["path_keep"][:, col],
                self.delay_outlier[:, col],
                color,
                label,
            )
            self.plot_click_points(
                self.ax_fft,
                "fft",
                col,
                t,
                self.event_arrays["fft_residuals_hz"][:, col] / 1e3,
                self.event_arrays["fft_keep"][:, col],
                self.fft_outlier[:, col],
                color,
                label,
            )
        self.ax_delay.set_title("Delay residuals")
        self.ax_delay.set_ylabel("m")
        self.ax_delay.set_ylim(-100, 100)
        self.ax_fft.set_title("Doppler residuals")
        self.ax_fft.set_ylabel("Beat residual (kHz)")
        self.ax_fft.set_xlabel("Time since t0 (s)")
        self.ax_fft.set_ylim(
            total_path_rate_mps_to_beat_residual_khz(1000.0),
            total_path_rate_mps_to_beat_residual_khz(-1000.0),
        )
        self.fft_secondary_axis = self.ax_fft.secondary_yaxis(
            "right",
            functions=(beat_residual_khz_to_total_path_rate_mps, total_path_rate_mps_to_beat_residual_khz),
        )
        self.fft_secondary_axis.set_ylabel("Equivalent total-path-rate residual (m/s)")

    def plot_click_points(self, ax, kind, col, t, y, keep, manual_outlier, color, label):
        finite = np.isfinite(t) & np.isfinite(y)
        groups = [
            (finite & keep & ~manual_outlier, "o", color, color, 0.85, label if kind == "delay" else None),
            (finite & ~keep & ~manual_outlier, "o", "none", color, 0.30, None),
            (finite & manual_outlier, "x", "crimson", "crimson", 0.95, None),
        ]
        for mask, marker, face, edge, alpha, legend_label in groups:
            indices = np.flatnonzero(mask)
            if not indices.size:
                continue
            artist = ax.scatter(
                t[indices],
                y[indices],
                s=34 if marker == "x" else 24,
                marker=marker,
                facecolors=face,
                edgecolors=edge,
                linewidths=0.9,
                alpha=alpha,
                picker=6,
                label=legend_label,
            )
            self.pick_map[artist] = (kind, col, indices)

    def status_string(self, row, summary):
        lines = [
            f"event: {row['event_id']} | status: {summary.get('status', '')}",
            f"model: {summary.get('dynamical_model', '')}",
            f"fallback: {summary.get('fallback_reason', '')}",
            f"bad: {summary.get('bad_fit_detected', '')}",
            f"bad reasons: {summary.get('bad_fit_reasons', '')}",
            f"recovery: {summary.get('bad_fit_recovery_step', '')}",
            f"coincident rows: {summary.get('n_coincident_delay_constraint_rows', '')}",
            f"coincident weight: {summary.get('coincident_delay_weight', self.coincident_delay_weight)}",
            f"fit mode: {summary.get('fit_mode', '')}",
            f"path rms m: {float(summary.get('rms_total_path_residual_m', np.nan)):.3g}",
            f"path mean |m|: {float(summary.get('mean_abs_total_path_residual_m', np.nan)):.3g}",
            f"fft rms Hz: {float(summary.get('rms_fft_residual_hz', np.nan)):.3g}",
            f"fft mean |Hz|: {float(summary.get('mean_abs_fft_residual_hz', np.nan)):.3g}",
            f"manual delay out: {int(np.count_nonzero(self.delay_outlier)) if self.delay_outlier is not None else 0}",
            f"manual fft out: {int(np.count_nonzero(self.fft_outlier)) if self.fft_outlier is not None else 0}",
            "keys: left/right browse | lasso bottom panel = included points | click toggles | F delay-only fit | G/R joint fit | C clear masks | q closes",
        ]
        return " | ".join(str(line) for line in lines if str(line))

    def save_current_masks(self):
        row = self.current()
        save_manual_masks(
            self.manual_outlier_h5,
            row["event_id"],
            self.event_arrays["time_ns"],
            self.delay_outlier,
            self.fft_outlier,
        )

    def on_lasso(self, kind, vertices):
        if self.event_arrays is None or self.delay_outlier is None or self.fft_outlier is None:
            return
        if vertices is None or len(vertices) < 3:
            return
        selector = MplPath(vertices)
        t = np.asarray(self.event_arrays["t_rel_s"], dtype=np.float64)
        if kind == "delay":
            y = np.asarray(self.event_arrays["path_residuals_m"], dtype=np.float64)
            finite = np.isfinite(t[:, None]) & np.isfinite(y)
            points = np.column_stack([np.repeat(t, 3), y.reshape(-1)])
            inside = selector.contains_points(points).reshape(y.shape) & finite
            self.delay_outlier[finite] = ~inside[finite]
            n_in = int(np.count_nonzero(inside))
        else:
            y = np.asarray(self.event_arrays["fft_residuals_hz"], dtype=np.float64) / 1e3
            finite = np.isfinite(t[:, None]) & np.isfinite(y)
            points = np.column_stack([np.repeat(t, 3), y.reshape(-1)])
            inside = selector.contains_points(points).reshape(y.shape) & finite
            self.fft_outlier[finite] = ~inside[finite]
            n_in = int(np.count_nonzero(inside))
        self.save_current_masks()
        self.message = f"{self.current()['event_id']}: lasso kept {n_in} {kind} measurements"
        self.load_current()

    def on_pick(self, event):
        if event.artist not in self.pick_map or self.event_arrays is None:
            return
        kind, col, indices = self.pick_map[event.artist]
        if not len(event.ind):
            return
        row_idx = int(indices[int(event.ind[0])])
        if kind == "delay":
            self.delay_outlier[row_idx, col] = ~self.delay_outlier[row_idx, col]
            state = "outlier" if self.delay_outlier[row_idx, col] else "not outlier"
        else:
            self.fft_outlier[row_idx, col] = ~self.fft_outlier[row_idx, col]
            state = "outlier" if self.fft_outlier[row_idx, col] else "not outlier"
        self.save_current_masks()
        row = self.current()
        self.message = f"{row['event_id']}: {kind} {SITE_LABELS[col]} row {row_idx} -> {state}"
        self.load_current()

    def goto(self, index):
        self.index = int(np.clip(index, 0, len(self.rows) - 1))
        self.load_current()

    def on_key(self, event):
        if event.key == "right":
            self.goto(self.index + 1)
        elif event.key == "left":
            self.goto(self.index - 1)
        elif event.key in {"f", "F"}:
            self.refit_current("delay-only")
        elif event.key in {"g", "G", "r", "R"}:
            self.refit_current("joint")
        elif event.key in {"c", "C"}:
            if self.event_arrays is not None:
                self.delay_outlier[:, :] = False
                self.fft_outlier[:, :] = False
                self.save_current_masks()
                self.message = f"{self.current()['event_id']}: cleared manual masks"
                self.load_current()
        elif event.key == "q":
            plt.close(self.fig)

    def refit_command(self, row, fit_mode):
        script = Path(__file__).with_name("fit_event_joint_delay_doppler_fft.py")
        return [
            sys.executable,
            str(script),
            "--event-id",
            row["event_id"],
            "--snr-min-db",
            str(self.snr_min_db),
            "--output-base",
            str(row["base"]),
            "--seed-from-existing-h5",
            str(row["h5"]),
            "--fit-mode",
            str(fit_mode),
            "--manual-outlier-h5",
            str(self.manual_outlier_h5),
            "--random-initial-guesses",
            str(self.random_initial_guesses),
            "--random-seed",
            str(int(time.time_ns() % (2**32))),
            "--force-model-reevaluation",
            "--coincident-delay-weight",
            str(self.coincident_delay_weight),
            *self.extra_fit_args,
        ]

    def run_refit(self, row, fit_mode):
        cmd = self.refit_command(row, fit_mode)
        proc = subprocess.run(cmd, cwd=Path(__file__).parent, text=True, capture_output=True)
        return proc.returncode, proc.stdout, proc.stderr, cmd, fit_mode

    def refit_current(self, fit_mode="joint"):
        if self.pending is not None and not self.pending.done():
            self.message = "Refit already running."
            self.load_current()
            return
        row = dict(self.current())
        label = "delay-only" if fit_mode == "delay-only" else "delay+beat"
        self.message = f"Refitting {row['event_id']} ({label}, CV vs SR model selection)..."
        self.pending = self.executor.submit(self.run_refit, row, fit_mode)
        self.load_current()

    def poll_refit(self):
        if self.pending is None or not self.pending.done():
            return True
        try:
            returncode, stdout, stderr, cmd, fit_mode = self.pending.result()
            if returncode == 0:
                self.message = f"Refit finished ({fit_mode}): {self.current()['event_id']}"
            else:
                self.message = (
                    f"Refit failed ({returncode}): {self.current()['event_id']}\n"
                    f"{stderr.strip()[:500]}"
                )
            if stdout.strip():
                print(stdout.strip(), flush=True)
            if stderr.strip():
                print(stderr.strip(), flush=True)
            print(" ".join(cmd), flush=True)
        except Exception as exc:
            self.message = f"Refit exception: {exc}"
        self.pending = None
        self.load_current()
        return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--event-id", default=None)
    parser.add_argument("--snr-min-db", type=float, default=DEFAULT_SNR_MIN_DB)
    parser.add_argument("--manual-outlier-h5", type=Path, default=None)
    parser.add_argument("--random-initial-guesses", type=int, default=24)
    parser.add_argument("--coincident-delay-weight", type=float, default=joint_fit.DEFAULT_COINCIDENT_DELAY_WEIGHT)
    parser.add_argument(
        "--fit-arg",
        action="append",
        default=[],
        help="Extra argument token passed to fit_event_joint_delay_doppler_fft.py. Repeat for multiple tokens.",
    )
    return parser.parse_args()


def main():
    reserve_keybindings()
    args = parse_args()
    browser = EventPlotBrowser(
        args.results_dir,
        args.event_id,
        args.snr_min_db,
        args.fit_arg,
        manual_outlier_h5=args.manual_outlier_h5,
        random_initial_guesses=args.random_initial_guesses,
        coincident_delay_weight=args.coincident_delay_weight,
    )
    plt.show()
    browser.executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
