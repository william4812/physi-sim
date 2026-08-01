#!/usr/bin/env python3
"""Anisotropic finite-volume reference solve of a ChipLayout JSON.

WHY A SECOND IMPLEMENTATION EXISTS
----------------------------------
Two purposes, both deliberate.

1. Capability, today. ElectroThermal3D is currently cubic-only (a single n and a
   single L), and a realistic 2.5D package is roughly 6 mm laterally by 0.5 mm
   tall. That aspect ratio is not cosmetic: the lateral healing length of the
   interposer is lam = sqrt(k t R''_vertical) ~ 670 um, so GPU->HBM crosstalk only
   resolves when the domain spans several lam. This script solves the same JSON
   the C++ app reads, at the correct aspect ratio.

2. Cross-implementation verification, later. When the C++ solver accepts
   (nx,ny,nz) and (Lx,Ly,Lz), it must reproduce these temperatures. Two
   independent implementations of the same discretisation agreeing is a genuine
   verification signal -- distinct from the MMS order-of-accuracy tests, which
   verify one implementation against the equations, and complementary to them.

DISCRETISATION -- identical in form to the C++ solver, so agreement is meaningful:
  * cell-centred finite volume, unknown is the CELL AVERAGE
  * interior face conductivity is the HARMONIC mean (series resistance: the flux
    crosses two half-cells in series, so resistances add and conductivities
    combine harmonically). Exact at a face-aligned material interface.
  * Robin faces use half-cell conduction in series with the film:
    U = A / (dz/(2k) + 1/h)
  * adiabatic faces contribute nothing to the diagonal -- no conductance path

Usage:  python3 python/scripts/solve_layout_reference.py chip_layout.json
"""
import argparse
import json

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

UM, W_CM2, C2K = 1e-6, 1e4, 273.15


def sample(cfg):
    """Sample the layout onto cell centres. Regions apply IN ORDER, later wins."""
    Lx, Ly, Lz = [v * UM for v in cfg["domain"]["size_um"]]
    nx, ny, nz = cfg["domain"]["cells"]
    hx, hy, hz = Lx / nx, Ly / ny, Lz / nz
    X, Y, Z = np.meshgrid((np.arange(nx) + .5) * hx / UM,
                          (np.arange(ny) + .5) * hy / UM,
                          (np.arange(nz) + .5) * hz / UM, indexing="ij")

    k = np.zeros((nx, ny, nz))
    qv = np.zeros((nx, ny, nz))
    tag = np.zeros((nx, ny, nz), dtype=int)
    names = []
    for r in cfg["regions"]:
        m = np.ones_like(k, dtype=bool)
        for key, arr in (("x_um", X), ("y_um", Y), ("z_um", Z)):
            if key in r:
                lo, hi = r[key]
                m &= (arr >= lo) & (arr < hi)      # half-open, tiles without gaps
        k[m] = cfg["materials"][r["material"]]["k_w_mk"]
        names.append(r["name"])
        tag[m] = len(names)
        if "power_w_cm2" in r:
            t = (r["z_um"][1] - r["z_um"][0]) * UM
            qv[m] = r["power_w_cm2"] * W_CM2 / t   # W/m^2 / m = W/m^3
    if k.min() <= 0:
        raise SystemExit("layout has a hole: some cell has no material")
    return (nx, ny, nz), (hx, hy, hz), k, qv, tag, names


def solve(cfg, dims, h, k, qv):
    nx, ny, nz = dims
    hx, hy, hz = h
    N = nx * ny * nz
    idx = np.arange(N).reshape(nx, ny, nz)
    diag = np.zeros((nx, ny, nz))
    b = qv * hx * hy * hz                          # [W/m^3][m^3] = W
    rows, cols, vals = [], [], []

    for ax, (n_, d_, A_) in enumerate([(nx, hx, hy * hz), (ny, hy, hx * hz), (nz, hz, hx * hy)]):
        lo = [slice(None)] * 3; lo[ax] = slice(0, n_ - 1)
        hi = [slice(None)] * 3; hi[ax] = slice(1, n_)
        kP, kN = k[tuple(lo)], k[tuple(hi)]
        g = 2 * kP * kN / (kP + kN) * A_ / d_       # harmonic mean * A/d
        P, Q = idx[tuple(lo)].ravel(), idx[tuple(hi)].ravel()
        rows += [P, Q]; cols += [Q, P]; vals += [-g.ravel(), -g.ravel()]
        diag[tuple(lo)] += g; diag[tuple(hi)] += g

    for face, bc in cfg["boundaries"].items():
        if bc["type"] == "adiabatic":
            continue                               # no conductance path; diagonal untouched
        if bc["type"] != "robin":
            raise SystemExit(f"reference solver handles adiabatic and robin only (got {bc['type']})")
        ax, end = {"x": 0, "y": 1, "z": 2}[face[0]], face.split("_")[1]
        sl = [slice(None)] * 3; sl[ax] = (0 if end == "low" else -1)
        A_ = [hy * hz, hx * hz, hx * hy][ax]
        d_ = [hx, hy, hz][ax]
        kw = k[tuple(sl)]
        U = A_ / (d_ / (2 * kw) + 1.0 / bc["h_w_m2k"])   # half-cell in series with film
        diag[tuple(sl)] += U
        b[tuple(sl)] += U * (bc["t_inf_c"] + C2K)

    rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(diag.ravel())
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(N, N)).tocsr()
    return spsolve(A, b.ravel()).reshape(nx, ny, nz) - C2K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout")
    ap.add_argument("--crosstalk", action="store_true",
                    help="isolate each region's contribution by zeroing the others")
    args = ap.parse_args()
    cfg = json.load(open(args.layout))

    dims, h, k, qv, tag, names = sample(cfg)
    T = solve(cfg, dims, h, k, qv)
    Lx, Lz = cfg["domain"]["size_um"][0] * UM, cfg["domain"]["size_um"][2] * UM

    print(f"grid {dims[0]}x{dims[1]}x{dims[2]} = {np.prod(dims)} cells   "
          f"dx={h[0]/UM:.0f} dy={h[1]/UM:.0f} dz={h[2]/UM:.1f} um   aspect {Lx/Lz:.1f}:1\n")
    print(f"  {'region':<12s} {'peak T':>9s} {'limit':>8s} {'margin':>9s}")
    for i, nm in enumerate(names, start=1):
        m = tag == i
        if not m.any():
            continue
        mat = cfg["materials"][cfg["regions"][i - 1]["material"]]
        lim = mat.get("t_limit_c")
        peak = T[m].max()
        margin = f"{lim - peak:+8.2f}" if lim else "       -"
        print(f"  {nm:<12s} {peak:8.2f}C {('%.0fC' % lim) if lim else '   -':>8s} {margin:>9s}")

    if args.crosstalk:
        powered = [i for i, r in enumerate(cfg["regions"], start=1) if "power_w_cm2" in r]
        print("\n  crosstalk decomposition (temperature rise above coolant):")
        t_inf = cfg["boundaries"]["z_high"]["t_inf_c"]
        for tgt in powered:
            m = tag == tgt
            alone = solve(cfg, dims, h, k, np.where(m, qv, 0.0))
            self_r = alone[m].max() - t_inf
            cross_r = T[m].max() - alone[m].max()
            tot = self_r + cross_r
            print(f"    {names[tgt-1]:<12s} self {self_r:6.2f} K   "
                  f"neighbours {cross_r:6.2f} K   ({100*cross_r/tot:3.0f}% from neighbours)")


if __name__ == "__main__":
    main()
