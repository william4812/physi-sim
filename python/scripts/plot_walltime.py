#!/usr/bin/env python3
"""Wall-time figures — two views, auto-detected from the CSV columns.

SCALING (per-grid profiling CSVs, columns: solver, grid_nx, wall_time_ms):
    wall time vs grid size (log-log), one line per solver. Every solver present
    shows up automatically (JacobiCPU, TDMACPU, JacobiGPU, ...).

RESIDENCY / COMPARISON (cmp_timing.csv, columns: variant, grid_n, wall_time_ms):
    grouped bars per grid, one bar per variant, log-y — the NoVram-vs-Vram
    punchline. Variant strings are read from the file, nothing hard-coded.

Usage:
    python3 plot_walltime.py --dir build --out docs/figures/wall_time.png
    python3 plot_walltime.py build/cmp_timing.csv --out docs/figures/wall_time_residency.png

Pass per-grid profiling CSVs OR cmp_timing.csv (one schema per call).
Only deps: pandas + matplotlib (numpy via matplotlib).
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_scaling(df, out, title):
    df = df.sort_values(["solver", "grid_nx"])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for solver, g in df.groupby("solver"):
        ax.loglog(g["grid_nx"], g["wall_time_ms"], marker="o", linewidth=1.7, label=solver)
    sizes = sorted(df["grid_nx"].unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Grid size  N  (N x N grid)")
    ax.set_ylabel("Wall time to convergence (ms)")
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(df.pivot_table(index="grid_nx", columns="solver",
                         values="wall_time_ms").to_string(float_format=lambda v: f"{v:.1f}"))

def variant_color(v):
    # Hue = algorithm, shade = residency — read from the name so the plotter
    # stays data-driven (no hardcoded variant list). "novram" is checked
    # before "vram" since it contains it as a substring.
    name = v.lower()
    cmap = (plt.cm.Blues if "jacobi" in name
            else plt.cm.Oranges if "tdma" in name
            else plt.cm.Greys)
    shade = 0.65 if "novram" in name else 0.85 if "vram" in name else 0.48
    return cmap(shade)

def variant_sort_key(v):
    name = v.lower()
    algo  = 0 if "jacobi" in name else 1                          # Jacobi block, then TDMA
    resid = 1 if "novram" in name else 2 if "vram" in name else 0  # CPU < NoVram < Vram
    return (algo, resid)

def plot_comparison(df, out, title):
    grids = sorted(df["grid_n"].unique())
    variants = sorted(dict.fromkeys(df["variant"]), key=variant_sort_key)
    #variants = list(dict.fromkeys(df["variant"]))   # preserve first-seen order
    x = np.arange(len(grids))
    w = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for i, v in enumerate(variants):
        vals = [df[(df["variant"] == v) & (df["grid_n"] == g)]["wall_time_ms"].mean()
                for g in grids]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=v, color=variant_color(v)) 
        for b, val in zip(bars, vals):
            if val == val:  # skip NaN
                ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.0f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}x{g}" for g in grids])
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Wall time to convergence (ms)")
    ax.set_title(title)
    ax.grid(True, which="both", axis="y", linewidth=0.4, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(df.pivot_table(index="grid_n", columns="variant",
                         values="wall_time_ms").to_string(float_format=lambda v: f"{v:.1f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*",
                    help="profiling CSVs, or cmp_timing.csv (default: --dir/*_profiling_results.csv)")
    ap.add_argument("--dir", default=".")
    ap.add_argument("--out", default="wall_time.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    files = args.files or sorted(glob.glob(str(Path(args.dir) / "*_profiling_results.csv")))
    if not files:
        raise SystemExit("no input CSVs found")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cols = set(df.columns)

    if {"solver", "grid_nx", "wall_time_ms"} <= cols:
        plot_scaling(df, args.out, args.title or "Wall time to convergence vs grid size")
    elif {"variant", "grid_n", "wall_time_ms"} <= cols:
        plot_comparison(df, args.out, args.title or "Wall time by variant (CPU / GPU NoVram / GPU Vram)")
    else:
        raise SystemExit(f"unrecognized columns: {sorted(cols)}")


if __name__ == "__main__":
    main()
