#!/usr/bin/env python3
"""
scripts/generate_benchmark_report.py
Generates two new README figures:
  1. convergence_with_gpu.png  — CPU curves + GPU final point
  2. cpu_vs_gpu_benchmark.png  — iteration count and wall time bar charts

USAGE:
    cd build
    ctest --output-on-failure          # produces jacobi/tdma CSV files
    python3 ../python/scripts/generate_benchmark_report.py \
        --output-dir . \
        --save-dir ../docs/figures

REQUIRES profiling_results.csv from ProfilingHarness::writeCSV().
If not present, GPU annotation is skipped gracefully.
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR.parent))

from physi_analytics import loaders, plotting

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report figures.")
    parser.add_argument("--output-dir", required=True, help="Directory with CSV files (build/)")
    parser.add_argument("--save-dir",   required=True, help="Directory to save figures (docs/figures/)")
    args = parser.parse_args()

    out_path  = Path(args.output_dir)
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # ── Load convergence histories ─────────────────────────────────────────────
    print("[1/3] Loading convergence CSVs...")
    jacobi = loaders.load_convergence_csv(out_path / "jacobi_convergence.csv")
    tdma   = loaders.load_convergence_csv(out_path / "tdma_convergence.csv")

    # ── Load ProfilingHarness results (optional — GPU data) ───────────────────
    profiling_path = out_path / "profiling_results.csv"
    profiling = None
    if profiling_path.exists():
        profiling = pd.read_csv(profiling_path)
        profiling.columns = [c.strip() for c in profiling.columns]
        print(f"[2/3] Loaded profiling data: {len(profiling)} solvers")
    else:
        print("[2/3] profiling_results.csv not found — GPU annotation skipped")
        print("      To generate: add harness.writeCSV('profiling_results.csv') to main.cpp")

    # ── Figure 1: convergence curves + GPU point ──────────────────────────────
    plotting.plot_convergence_with_gpu(
        jacobi, tdma, profiling,
        save_path / "convergence_with_gpu.png"
    )

    # ── Figure 2: benchmark bar charts ────────────────────────────────────────
    if profiling is not None:
        plotting.plot_benchmark_bars(
            profiling,
            save_path / "cpu_vs_gpu_benchmark.png"
        )
        print("[3/3] Benchmark bar chart saved.")
    else:
        print("[3/3] Skipped benchmark bars — no profiling_results.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    ratio = len(tdma) / len(jacobi)
    print(f"\n── Convergence summary ─────────────────────────────────────")
    print(f"  JacobiCPU : {len(jacobi):5,} iters  "
          f"final = {jacobi['Residual'].iloc[-1]:.3e}")
    print(f"  TDMACPU   : {len(tdma):5,} iters  "
          f"final = {tdma['Residual'].iloc[-1]:.3e}")
    print(f"  TDMA/Jacobi ratio: {ratio:.3f}  ({1/ratio:.2f}× fewer iterations)")
    if profiling is not None:
        gpu = profiling[profiling["solver_name"] == "JacobiGPU"]
        cpu = profiling[profiling["solver_name"] == "JacobiCPU"]
        if not gpu.empty and not cpu.empty:
            speedup = float(cpu["wall_time_ms"].iloc[0]) / float(gpu["wall_time_ms"].iloc[0])
            print(f"  GPU speedup: {speedup:.2f}×  "
                  f"({float(cpu['wall_time_ms'].iloc[0]):.1f}ms CPU → "
                  f"{float(gpu['wall_time_ms'].iloc[0]):.1f}ms GPU)")
    print("─────────────────────────────────────────────────────────────")

if __name__ == "__main__":
    main()
