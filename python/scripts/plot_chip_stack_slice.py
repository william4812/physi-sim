#!/usr/bin/env python3
"""Render the mid-plane temperature + heat-flux slice from a thermal3d .vti file.

Fallback / automation for the ParaView workflow: same money shot (hotspot + TIM
bottleneck) as a PNG, no ParaView dependency. Companion to plot_thermal_map.py.

Usage:
    python3 python/scripts/plot_chip_stack_slice.py build/chip_stack.vti \
        --out docs/figures/chip_stack_slice.png --si-thickness-um 300
"""
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_vti(path):
    root = ET.parse(path).getroot()
    img = root.find(".//ImageData")
    n = int(img.get("WholeExtent").split()[1])
    dx = float(img.get("Spacing").split()[0])
    arrays = {}
    for d in root.iter("DataArray"):
        vals = np.fromstring(d.text, sep=" ")
        comps = int(d.get("NumberOfComponents", "1"))
        arrays[d.get("Name")] = (vals.reshape(n, n, n) if comps == 1
                                 else vals.reshape(n, n, n, comps))
    return n, dx, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vti")
    ap.add_argument("--out", default="chip_stack_slice.png")
    ap.add_argument("--si-thickness-um", type=float, default=None)
    args = ap.parse_args()

    n, dx, a = load_vti(args.vti)
    T = a["temperature_C"] # Re-purposed as material ID map
    q = a.get("heat_flux", None)
    
    jm = n // 2
    um = 1e6 * dx
    xs = (np.arange(n) + 0.5) * um
    zs = (np.arange(n) + 0.5) * um
    Tsl = T[:, jm, :]                 # [z, x]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    im = ax.pcolormesh(xs, zs, Tsl, cmap="tab10", shading="auto")
    fig.colorbar(im, ax=ax).set_label("Material ID")

    if q is not None:
        qxz = q[:, jm, :, 0]
        qzz = q[:, jm, :, 2]
        st = max(1, n // 12)
        ax.quiver(xs[::st], zs[::st], qxz[::st, ::st], qzz[::st, ::st],
                  color="white", alpha=0.75, width=0.004)

    if args.si_thickness_um is not None:
        ax.axhline(args.si_thickness_um, color="cyan", ls="--", lw=1.2)
        ax.text(xs[-1] * 0.99, args.si_thickness_um, " Si | TIM ",
                color="cyan", va="bottom", ha="right", fontsize=9)

    ax.set_xlabel("lateral x  [$\\mu$m]")
    ax.set_ylabel("through-stack z  [$\\mu$m]   (die backside 0 $\\to$ coldplate)")
    ax.set_title("GPU-HBM Modular Package Layout")
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"wrote {args.out}   ID range {Tsl.min():.1f} .. {Tsl.max():.1f}")

    ax.set_xlabel("lateral x  [$\\mu$m]")
    ax.set_ylabel("through-stack z  [$\\mu$m]   (die backside 0 $\\to$ coldplate)")
    ax.set_title("Mid-plane temperature and heat flux  (from solver .vti)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"wrote {args.out}   T range {Tsl.min():.1f} .. {Tsl.max():.1f} C")


if __name__ == "__main__":
    main()
