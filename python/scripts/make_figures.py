#!/usr/bin/env python3
"""One-run figure pipeline for physi-sim.

Sequences the three standalone plotters over a benchmark output directory and
writes every figure to a save directory.

Design (low coupling, high cohesion):
  * Each plot_*.py stays a self-contained, single-purpose tool with its own CLI.
    This orchestrator never imports their internals — it calls them across the
    CLI boundary, the loosest possible coupling.
  * This file is the ONLY place that knows the benchmark's filename convention
    and the figure recipe (the composition root).
  * A figure whose inputs are missing is skipped, not fatal — a partial
    benchmark still produces whatever it can.

Usage:
    python3 make_figures.py --data build --out docs/figures
    python3 make_figures.py --data build --out docs/figures --grid 100 --tol 1e-7
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent   # plot_*.py live next to this file


def have(*paths) -> bool:
    return all(Path(p).exists() for p in paths)


def run(script: str, *args) -> None:
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    print("   $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate all physi-sim figures in one run.")
    ap.add_argument("--data", default="build", help="benchmark output dir (CSV + VTK)")
    ap.add_argument("--out", default="docs/figures", help="where to write PNGs")
    ap.add_argument("--grid", type=int, default=100,
                    help="reference grid N for the convergence + thermal figures")
    ap.add_argument("--tol", type=float, default=1e-7,
                    help="stopping tolerance drawn on the convergence figure")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    N = args.grid

    done, skipped, failed = [], [], []

    def attempt(name, inputs, script, *script_args):
        missing = [str(p) for p in inputs if not Path(p).exists()]
        if missing:
            skipped.append((name, Path(missing[0]).name))
            print(f"[skip] {name}  (missing {Path(missing[0]).name})")
            return
        try:
            print(f"[run ] {name}")
            run(script, *script_args)
            done.append(name)
        except subprocess.CalledProcessError as e:
            failed.append((name, e.returncode))
            print(f"[FAIL] {name}  (exit {e.returncode})")

    # 1) Convergence — CPU pair at the reference grid
    jac = data / f"JacobiCPU_{N}x{N}_convergence.csv"
    tdma = data / f"TDMACPU_{N}x{N}_convergence.csv"
    attempt("convergence", [jac, tdma], "plot_convergence.py",
            f"{jac}:Jacobi", f"{tdma}:TDMA",
            "--tol", args.tol, "--out", out / f"convergence_{N}x{N}.png")

    # 2a) Wall time — scaling across all grids present
    prof = sorted(data.glob("*_profiling_results.csv"))
    if prof:
        attempt("wall_time", [prof[0]], "plot_walltime.py",
                "--dir", data, "--out", out / "wall_time.png")
    else:
        skipped.append(("wall_time", "*_profiling_results.csv"))
        print("[skip] wall_time  (no *_profiling_results.csv)")

    # 2b) Wall time — NoVram vs Vram residency
    attempt("wall_time_residency", [data / "cmp_timing.csv"],
            "plot_walltime.py", data / "cmp_timing.csv",
            "--out", out / "wall_time_residency.png")

    # 3a) Thermal field — single converged map at the reference grid
    jvtk = data / f"cmp_cpu_jacobi_{N}.vtk"
    attempt("thermal_field", [jvtk], "plot_thermal_map.py",
            jvtk, "--out", out / "thermal_field.png")

    # 3b) Thermal consistency — every variant present, side by side
    variants = [
        (data / f"cmp_cpu_jacobi_{N}.vtk", "Jacobi CPU"),
        (data / f"cmp_cpu_tdma_{N}.vtk", "TDMA CPU"),
        (data / f"cmp_gpu_jacobi_novram_{N}.vtk", "GPU NoVram"),
        (data / f"cmp_gpu_jacobi_vram_{N}.vtk", "GPU Vram"),
    ]
    present = [(p, lbl) for p, lbl in variants if p.exists()]
    if len(present) >= 2:
        attempt("thermal_compare", [p for p, _ in present], "plot_thermal_map.py",
                *[f"{p}:{lbl}" for p, lbl in present],
                "--out", out / f"thermal_compare_{N}.png")
    else:
        skipped.append(("thermal_compare", "need >=2 cmp_*.vtk"))
        print("[skip] thermal_compare  (need >=2 cmp_*.vtk)")

    # Manifest
    print(f"\n=== {len(done)} figure(s) written to {out} ===")
    for d in done:
        print(f"  ok    {d}")
    for name, why in skipped:
        print(f"  skip  {name}  ({why})")
    for name, code in failed:
        print(f"  FAIL  {name}  (exit {code})")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
