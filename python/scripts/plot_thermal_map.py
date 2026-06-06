#!/usr/bin/env python3
"""Thermal map(s) from STRUCTURED_POINTS VTK — no pyvista.

Parses the ASCII VTK your VTKWriter::write_2d emits (DIMENSIONS nx ny 1,
SCALARS Temperature double) and renders a heatmap. Pass several files to get a
side-by-side row with a shared color scale (e.g. the four cmp_*.vtk variants —
they should look identical, which is your cross-solver consistency check).

Usage:
    python3 plot_thermal_map.py build/cmp_cpu_jacobi_100.vtk --out docs/figures/thermal_field.png
    python3 plot_thermal_map.py build/cmp_cpu_jacobi_100.vtk:"Jacobi CPU" \
                                build/cmp_cpu_tdma_100.vtk:"TDMA CPU" \
                                --out docs/figures/thermal_compare.png

Only deps: numpy + matplotlib.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_structured_points(path):
    """Return the field as a (ny, nx) array (VTK point order is x-fastest)."""
    nx = ny = None
    tokens = []
    after_lut = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.upper().startswith("DIMENSIONS"):
                p = s.split()
                nx, ny = int(p[1]), int(p[2])
            elif s.upper().startswith("LOOKUP_TABLE"):
                after_lut = True
            elif after_lut and s:
                tokens.extend(s.split())
    if nx is None:
        raise ValueError(f"{path}: no DIMENSIONS line found")
    vals = np.array(tokens[:nx * ny], dtype=float)
    return vals.reshape(ny, nx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("maps", nargs="+", help="file[:label] per VTK")
    ap.add_argument("--out", default="thermal_field.png")
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--transpose", action="store_true",
                    help="flip x/y if the hot boundary lands on the wrong edge")
    args = ap.parse_args()

    fields, labels = [], []
    for item in args.maps:
        path, _, label = item.partition(":")
        p = Path(path)
        if not p.exists():
            print(f"  skip (missing): {p}")
            continue
        F = read_structured_points(p)
        if args.transpose:
            F = F.T
        fields.append(F)
        labels.append(label or p.stem)

    if not fields:
        raise SystemExit("no VTK files read")

    vmin = min(F.min() for F in fields)
    vmax = max(F.max() for F in fields)
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.6), squeeze=False)
    im = None
    for ax, F, lab in zip(axes[0], fields, labels):
        im = ax.imshow(F, origin="lower", cmap=args.cmap,
                       vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(lab)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im, ax=list(axes[0]), shrink=0.85, label="Temperature")
    fig.suptitle("Converged temperature field"
                 + ("" if n == 1 else " — solver comparison"))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")

    print(f"wrote {args.out}")
    for F, lab in zip(fields, labels):
        print(f"  {lab:16s} {F.shape[1]}x{F.shape[0]}  "
              f"T in [{F.min():.3f}, {F.max():.3f}]")


if __name__ == "__main__":
    main()
