#!/usr/bin/env python3
"""Convergence comparison — normalized increment residual vs iteration.

Plots  ||T^k - T^{k-1}||_inf / ||T^1 - T^0||_inf  so every curve starts at 1
and terminates ON the shared stopping tolerance. That makes "iterations to the
same relative-increment threshold" an honest apples-to-apples comparison
(the raw-residual plot ends each curve at a different absolute floor, which
hides the fact that both stopped at the same *normalized* criterion).

NOTE: this is a per-ITERATION efficiency view, not wall-clock. Fewer iterations
does not mean faster — see the wall-time figure for the time story.

Usage:
    python3 plot_convergence.py \
        JacobiCPU_100x100_convergence.csv:Jacobi \
        TDMACPU_100x100_convergence.csv:TDMA \
        --tol 1e-7 --out convergence_100x100.png

Each positional arg is  path[:label]  (label defaults to the filename stem).
Add as many as you like (e.g. JacobiGPU_...csv:"Jacobi GPU").
CSV columns expected: Iteration,Residual  (absolute increment per step).
Only deps: pandas + matplotlib.
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / CI-friendly
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("series", nargs="+", help="path[:label] per solver")
    ap.add_argument("--tol", type=float, default=1e-7,
                    help="normalized stopping tolerance to draw (default: 1e-7)")
    ap.add_argument("--out", default="convergence.png")
    ap.add_argument("--title", default="Convergence — relative increment residual")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    summary = []

    for item in args.series:
        path, _, label = item.partition(":")
        p = Path(path)
        if not p.exists():
            print(f"  skip (missing): {p}")
            continue
        label = label or p.stem
        df = pd.read_csv(p)
        r0 = df["Residual"].iloc[0]
        norm = df["Residual"] / r0
        n = len(df)
        line, = ax.semilogy(df["Iteration"], norm, linewidth=1.7,
                            label=f"{label}  ({n:,} iters)")
        ax.scatter([df["Iteration"].iloc[-1]], [norm.iloc[-1]],
                   s=24, color=line.get_color(), zorder=3)
        summary.append((label, n, float(df["Residual"].iloc[-1])))

    ax.axhline(args.tol, color="0.4", linestyle="--", linewidth=1.0)
    ax.text(ax.get_xlim()[1], args.tol, f"  tol = {args.tol:g}",
            va="center", ha="left", fontsize=9, color="0.3")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Normalized increment  "
                  r"$\|T^{k}-T^{k-1}\|_\infty\,/\,\|T^{1}-T^{0}\|_\infty$")
    ax.set_title(args.title)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)

    print(f"wrote {args.out}")
    for label, n, final_abs in summary:
        print(f"  {label:12s} {n:>7,} iters   final |increment|={final_abs:.3e}")
    if len(summary) == 2:
        (l0, n0, _), (l1, n1, _) = summary
        if n0 and n1:
            if n0 < n1:
                print(f"  -> {l0} reaches tol in {n1/n0:.2f}x fewer iterations than {l1}")
            else:
                print(f"  -> {l1} reaches tol in {n0/n1:.2f}x fewer iterations than {l0}")


if __name__ == "__main__":
    main()
