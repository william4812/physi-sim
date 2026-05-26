#!/usr/bin/env python3
"""
scripts/generate_benchmark_report.py

Generates all README figures from physi-sim benchmark output.

FIGURES PRODUCED:
  jacobi_cpu_vs_gpu.png         — convergence + wall time + ms/iter (focused)
  thermal_field.png             — converged temperature field heatmap
  convergence_comparison.png    — Jacobi vs TDMA (regression test path)
  gpu_speedup_vs_grid_size.png  — speedup crossover across grid sizes

USAGE:
    cd build
    ctest --output-on-failure           # produces regression CSVs + VTK
    ./physi_sim                         # produces per-grid profiling CSVs
    python3 ../python/scripts/generate_benchmark_report.py \
        --output-dir . \
        --save-dir ../docs/figures
        --grid-size 100                 # reference grid for convergence panel
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR.parent))

from physi_analytics import loaders, plotting


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-dir",   required=True)
    parser.add_argument("--grid-size",  type=int, default=100,
                        help="Reference grid for convergence panel (default: 100)")
    args = parser.parse_args()

    out_path  = Path(args.output_dir)
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    N = args.grid_size

    print(f"Output dir: {out_path}")
    print(f"Save dir:   {save_path}")
    print(f"Reference grid: {N}×{N}\n")

    # ── Figure 1: Jacobi CPU vs GPU (the focused three-panel figure) ──────────
    print("[1/4] Generating Jacobi CPU vs GPU figure...")
    plotting.plot_jacobi_cpu_vs_gpu(
        out_path, N,
        save_path / "jacobi_cpu_vs_gpu.png")

    # ── Figure 2: Temperature field ───────────────────────────────────────────
    print("[2/4] Generating thermal field figure...")
    vtk_path = out_path / "jacobi_final_map.vtk"
    plotting.plot_thermal_field(
        vtk_path, N, N,
        save_path / "thermal_field.png")

    # ── Figure 3: Jacobi vs TDMA convergence (regression test path) ───────────
    print("[3/4] Generating Jacobi vs TDMA convergence...")
    jacobi_csv = out_path / "jacobi_convergence.csv"
    tdma_csv   = out_path / "tdma_convergence.csv"
    if jacobi_csv.exists() and tdma_csv.exists():
        jacobi = loaders.load_convergence_csv(jacobi_csv)
        tdma   = loaders.load_convergence_csv(tdma_csv)
        plotting.plot_residuals(jacobi, tdma,
                                save_path / "convergence_comparison.png")
        ratio = len(tdma) / len(jacobi)
        print(f"  JacobiCPU: {len(jacobi):,} iters  "
              f"final={jacobi['Residual'].iloc[-1]:.3e}")
        print(f"  TDMACPU:   {len(tdma):,} iters  "
              f"final={tdma['Residual'].iloc[-1]:.3e}")
        print(f"  Ratio: {ratio:.3f}  ({1/ratio:.2f}× fewer iters)")
    else:
        print("  Skipped — run ctest first to produce regression CSVs")

    # ── Figure 4: GPU speedup vs grid size ────────────────────────────────────
    print("[4/4] Generating GPU speedup vs grid size...")
    plotting.plot_speedup_vs_grid_size(
        out_path,
        save_path / "gpu_speedup_vs_grid_size.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Figures saved to {save_path} ───────────────────────────")
    for fig in sorted(save_path.glob("*.png")):
        print(f"  {fig.name}")
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
