#!/usr/bin/env python3
"""
scripts/generate_all_figures.py

Master pipeline — generates ALL README figures in phase order.
Each figure maps to one README section.

USAGE:
    cd build
    ctest --output-on-failure    # §2 + §3 data
    ./physi_sim                  # §4 data
    python3 ../python/scripts/generate_all_figures.py \
        --output-dir . \
        --save-dir ../docs/figures

OUTPUT:
    docs/figures/
      s2_thermal_field.png              — physics verification
      s3_convergence_jacobi_vs_tdma.png — algorithm comparison
      s4a_convergence_cpu_vs_gpu.png    — GPU correctness proof
      s4b_wall_time_analysis.png        — GPU performance analysis
      s4c_speedup_vs_grid_size.png      — PCIe crossover chart

Each filename is prefixed with its README section.
Figures are skipped gracefully when data is not yet available.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR.parent))

from physi_analytics import loaders, plotting

# ── Figure registry — maps section → (function, required_files) ──────────────
# Add new figures here only. Nothing else changes.
#
# Convention:
#   key:   "§N_filename"  → README section + output filename
#   fn:    plotting function to call
#   needs: list of files that must exist for this figure to run

def _build_registry(out: Path, save: Path, N: int) -> list:
    """
    Returns list of dicts: {section, filename, fn, needs, kwargs}
    Order = README section order.
    """
    return [
        # ── §2 Physics verification ───────────────────────────────────────
        {
            "section":  "§2",
            "filename": "s2_thermal_field.png",
            "fn":       plotting.plot_thermal_field,
            "needs":    [],   # graceful fallback if VTK missing
            "kwargs":   dict(
                vtk_path=out / "jacobi_final_map.vtk",
                grid_nx=N, grid_ny=N,
                save_path=save / "s2_thermal_field.png"),
        },

        # ── §3 Algorithm comparison — Jacobi vs TDMA (CPU) ───────────────
        {
            "section":  "§3",
            "filename": "s3_convergence_jacobi_vs_tdma.png",
            "fn":       _plot_jacobi_vs_tdma,
            "needs":    [out/"jacobi_convergence.csv", out/"tdma_convergence.csv"],
            "kwargs":   dict(
                out=out,
                save_path=save / "s3_convergence_jacobi_vs_tdma.png"),
        },

        # ── §4a GPU correctness — convergence curves ──────────────────────
        {
            "section":  "§4a",
            "filename": "s4a_convergence_cpu_vs_gpu.png",
            "fn":       plotting.plot_jacobi_convergence_grid,
            "needs":    [out/f"JacobiCPU_{N}x{N}_convergence.csv",
                         out/f"JacobiGPU_{N}x{N}_convergence.csv"],
            "kwargs":   dict(
                csv_dir=out,
                save_path=save / "s4a_convergence_cpu_vs_gpu.png"),
        },

        # ── §4b GPU performance — wall time analysis ──────────────────────
        {
            "section":  "§4b",
            "filename": "s4b_wall_time_analysis.png",
            "fn":       plotting.plot_wall_time_analysis,
            "needs":    [out/"50x50_profiling_results.csv"],
            "kwargs":   dict(
                csv_dir=out,
                save_path=save / "s4b_wall_time_analysis.png"),
        },

        # ── §4c GPU crossover — speedup vs grid size ──────────────────────
        {
            "section":  "§4c",
            "filename": "s4c_speedup_vs_grid_size.png",
            "fn":       plotting.plot_speedup_vs_grid_size,
            "needs":    [out/"100x100_profiling_results.csv"],
            "kwargs":   dict(
                csv_dir=out,
                save_path=save / "s4c_speedup_vs_grid_size.png"),
        },
    ]


def _plot_jacobi_vs_tdma(out: Path, save_path: Path) -> None:
    """Thin wrapper so registry can call with kwargs."""
    jacobi = loaders.load_convergence_csv(out / "jacobi_convergence.csv")
    tdma   = loaders.load_convergence_csv(out / "tdma_convergence.csv")
    plotting.plot_residuals(jacobi, tdma, save_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all README figures in phase order.")
    parser.add_argument("--output-dir", required=True,
                        help="build/ directory with CSV and VTK files")
    parser.add_argument("--save-dir",   required=True,
                        help="docs/figures/ output directory")
    parser.add_argument("--grid-size",  type=int, default=100,
                        help="Reference grid for §4a convergence (default: 100)")
    args   = parser.parse_args()
    out    = Path(args.output_dir)
    save   = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    N      = args.grid_size

    registry = _build_registry(out, save, N)

    print("physi-sim figure pipeline")
    print(f"  input:  {out}")
    print(f"  output: {save}")
    print(f"  ref grid: {N}×{N}\n")

    done    = []
    skipped = []

    for entry in registry:
        sec      = entry["section"]
        fname    = entry["filename"]
        fn       = entry["fn"]
        needs    = entry["needs"]
        kwargs   = entry["kwargs"]

        missing = [str(f) for f in needs if not Path(f).exists()]
        if missing:
            skipped.append((sec, fname, missing[0]))
            print(f"  [{sec}] SKIP  {fname}")
            print(f"         missing: {Path(missing[0]).name}")
            continue

        try:
            fn(**kwargs)
            done.append((sec, fname))
            print(f"  [{sec}] OK    {fname}")
        except Exception as e:
            skipped.append((sec, fname, str(e)))
            print(f"  [{sec}] FAIL  {fname}")
            print(f"         error: {e}")

    # ── Manifest ──────────────────────────────────────────────────────────
    print(f"\n── Generated {len(done)}/{len(registry)} figures ─────────────────")
    for sec, fname in done:
        print(f"  {sec}  docs/figures/{fname}")

    if skipped:
        print(f"\n── Skipped {len(skipped)} ────────────────────────────────────")
        for sec, fname, reason in skipped:
            print(f"  {sec}  {fname}")
            if "profiling" in reason or "convergence" in reason:
                print(f"         → run: cd build && ./physi_sim")
            elif "vtk" in reason.lower():
                print(f"         → run: ctest --output-on-failure -R Physics2DTest")
            elif "jacobi_convergence" in reason:
                print(f"         → run: ctest --output-on-failure -R Physics2DTest")

    print("\n── README image tags ────────────────────────────────────────────")
    for sec, fname in done:
        print(f"  ![]({('docs/figures/' + fname)})")
    print("─────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
