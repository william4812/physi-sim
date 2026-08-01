#!/usr/bin/env python3
"""Solve a ChipLayout JSON and write the temperature field to .vti for ParaView.
Reuses the same discretisation as solve_layout_reference.py (harmonic-mean FV,
half-cell Robin). CellData + x-fastest ordering => byte-compatible with the C++
io::VTKWriter, so the eventual anisotropic C++ solver can be checked against it.
Usage: python3 python/scripts/write_temperature_vti.py chip_layout.json out.vti
"""
import json, sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

UM, W_CM2, C2K = 1e-6, 1e4, 273.15
cfg = json.load(open(sys.argv[1]))
out = sys.argv[2] if len(sys.argv) > 2 else "chip_temperature.vti"
Lx, Ly, Lz = [v*UM for v in cfg['domain']['size_um']]
nx, ny, nz = cfg['domain']['cells']
hx, hy, hz = Lx/nx, Ly/ny, Lz/nz
X, Y, Z = np.meshgrid((np.arange(nx)+.5)*hx/UM, (np.arange(ny)+.5)*hy/UM,
                      (np.arange(nz)+.5)*hz/UM, indexing='ij')
k = np.zeros((nx, ny, nz)); qv = np.zeros((nx, ny, nz))
for r in cfg['regions']:
    m = np.ones_like(k, bool)
    for key, arr in (('x_um', X), ('y_um', Y), ('z_um', Z)):
        if key in r:
            lo, hi = r[key]; m &= (arr >= lo) & (arr < hi)
    k[m] = cfg['materials'][r['material']]['k_w_mk']
    if 'power_w_cm2' in r:
        t = (r['z_um'][1]-r['z_um'][0])*UM; qv[m] = r['power_w_cm2']*W_CM2/t

N = nx*ny*nz; idx = np.arange(N).reshape(nx, ny, nz)
diag = np.zeros((nx, ny, nz)); b = qv*hx*hy*hz
R, C, V = [], [], []
for ax, (n_, d_, A_) in enumerate([(nx, hx, hy*hz), (ny, hy, hx*hz), (nz, hz, hx*hy)]):
    lo = [slice(None)]*3; lo[ax] = slice(0, n_-1)
    hi = [slice(None)]*3; hi[ax] = slice(1, n_)
    kP, kN = k[tuple(lo)], k[tuple(hi)]
    g = 2*kP*kN/(kP+kN)*A_/d_
    P, Q = idx[tuple(lo)].ravel(), idx[tuple(hi)].ravel()
    R += [P, Q]; C += [Q, P]; V += [-g.ravel(), -g.ravel()]
    diag[tuple(lo)] += g; diag[tuple(hi)] += g
bc = cfg['boundaries']['z_high']; kw = k[:, :, -1]
U = hx*hy/(hz/(2*kw) + 1/bc['h_w_m2k'])
diag[:, :, -1] += U; b[:, :, -1] += U*(bc['t_inf_c']+C2K)
R.append(idx.ravel()); C.append(idx.ravel()); V.append(diag.ravel())
A = coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(N, N)).tocsr()
T = (spsolve(A, b.ravel()).reshape(nx, ny, nz) - C2K)

with open(out, 'w') as f:
    f.write('<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n')
    f.write(f'  <ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}" Origin="0 0 0" Spacing="{hx} {hy} {hz}">\n')
    f.write(f'    <Piece Extent="0 {nx} 0 {ny} 0 {nz}">\n      <CellData>\n')
    f.write('        <DataArray type="Float64" Name="temperature_C" format="ascii">\n          ')
    for l in range(nz):
        for j in range(ny):
            for i in range(nx):
                f.write(f'{T[i,j,l]:.9g} ')
    f.write('\n        </DataArray>\n      </CellData>\n    </Piece>\n  </ImageData>\n</VTKFile>\n')
print(f"wrote {out}   T {T.min():.2f} .. {T.max():.2f} C")
