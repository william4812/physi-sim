#!/usr/bin/env python3
"""thermal_bridge.py — measured GPU telemetry -> predicted INTERNAL thermal field.

The move a pure telemetry tool cannot make: NVML reports a single surface
temperature and the dissipated power. This feeds those as BOUNDARY CONDITIONS
into a 3D finite-volume conduction solve of the package and returns the field the
sensor cannot see -- the junction hotspot AND the lateral spread into neighbouring
memory (crosstalk). Same discretization as physi-sim's verified reference solver
(harmonic-mean faces, half-cell Robin), which matches the C++ solver to the digit.

Pipeline:  nvml_logger.py -> telemetry.csv -> [this] -> predicted_field.vti + summary

    python3 thermal_bridge.py telemetry.csv --vti predicted_field.vti
"""
import argparse, csv, sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

C2K = 273.15

def build_package(n_lat, n_z, L_lat, L_z, gpu_frac, k_die, k_hbm, k_tim, k_mold):
    """A GPU die centred in the lateral plane, HBM to either side, TIM cap on top.
    Returns per-cell conductivity k[nx,ny,nz] and a mask of the GPU die cells."""
    nx = ny = n_lat; nz = n_z
    k = np.full((nx, ny, nz), k_mold)
    # z layers: bottom 60% = die/hbm plane, top 40% = TIM
    z_tim0 = int(nz * 0.6)
    # lateral: centre square = GPU die, flanks = HBM
    lo = int(nx * (0.5 - gpu_frac/2)); hi = int(nx * (0.5 + gpu_frac/2))
    plane = np.full((nx, ny), k_hbm)          # HBM fills the device plane...
    plane[lo:hi, lo:hi] = k_die               # ...GPU die carved in the centre
    for z in range(z_tim0):
        k[:, :, z] = plane
    k[:, :, z_tim0:] = k_tim                   # TIM cap
    gpu_mask = np.zeros((nx, ny, nz), bool)
    gpu_mask[lo:hi, lo:hi, :z_tim0] = True
    return k, gpu_mask, (lo, hi, z_tim0)

def solve(k, q_v, h_top, Tinf_K, L):
    """FV solve of div(k grad T)+q_v=0, Robin on +z, adiabatic elsewhere.
    Identical scheme to physi-sim's reference solver."""
    nx, ny, nz = k.shape
    hx, hy, hz = L[0]/nx, L[1]/ny, L[2]/nz
    N = nx*ny*nz; idx = np.arange(N).reshape(nx,ny,nz)
    diag = np.zeros((nx,ny,nz)); b = q_v*hx*hy*hz
    R,C,V = [],[],[]
    for ax,(n_,d_,A_) in enumerate([(nx,hx,hy*hz),(ny,hy,hx*hz),(nz,hz,hx*hy)]):
        lo=[slice(None)]*3; lo[ax]=slice(0,n_-1)
        hi=[slice(None)]*3; hi[ax]=slice(1,n_)
        kP,kN=k[tuple(lo)],k[tuple(hi)]
        g=2*kP*kN/(kP+kN)*A_/d_                 # harmonic mean = series resistance
        P,Q=idx[tuple(lo)].ravel(),idx[tuple(hi)].ravel()
        R+=[P,Q];C+=[Q,P];V+=[-g.ravel(),-g.ravel()]
        diag[tuple(lo)]+=g; diag[tuple(hi)]+=g
    kw=k[:,:,-1]; U=hx*hy/(hz/(2*kw)+1.0/h_top)  # half-cell in series with film
    diag[:,:,-1]+=U; b[:,:,-1]+=U*Tinf_K
    R.append(idx.ravel());C.append(idx.ravel());V.append(diag.ravel())
    A=coo_matrix((np.concatenate(V),(np.concatenate(R),np.concatenate(C))),shape=(N,N)).tocsr()
    return spsolve(A,b.ravel()).reshape(nx,ny,nz)-C2K

def write_vti(path, T, L):
    nx,ny,nz=T.shape; hx,hy,hz=L[0]/nx,L[1]/ny,L[2]/nz
    with open(path,'w') as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n')
        f.write(f'  <ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}" Origin="0 0 0" Spacing="{hx} {hy} {hz}">\n')
        f.write(f'    <Piece Extent="0 {nx} 0 {ny} 0 {nz}">\n      <CellData>\n')
        f.write('        <DataArray type="Float64" Name="temperature_C" format="ascii">\n          ')
        for l in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f'{T[i,j,l]:.6g} ')
        f.write('\n        </DataArray>\n      </CellData>\n    </Piece>\n  </ImageData>\n</VTKFile>\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--vti", default=None, help="write predicted field for ParaView")
    ap.add_argument("--die-mm", type=float, default=8.0, help="GPU die edge (mm)")
    ap.add_argument("--h-top", type=float, default=5e4, help="coldplate coeff W/m^2K")
    args = ap.parse_args()

    with open(args.csv) as f:
        rows=list(csv.DictReader(f))
    if not rows: sys.exit("empty telemetry")
    peak=max(rows, key=lambda r: float(r["power_W"]))
    T_case=float(peak["temp_C"]); P=float(peak["power_W"])

    # --- geometry (representative package) ---
    L_lat=12e-3; L_z=0.5e-3; L=(L_lat,L_lat,L_z)
    n_lat=48; n_z=16
    gpu_frac=args.die_mm*1e-3/L_lat
    k,gpu_mask,_=build_package(n_lat,n_z,L_lat,L_z,gpu_frac,
                               k_die=150,k_hbm=20,k_tim=5,k_mold=0.8)

    # --- MEASURED boundary conditions from NVML ---
    # power -> volumetric source in the die; case temp -> coldplate T_inf
    die_vol=gpu_mask.sum()*(L_lat/n_lat)**2*(L_z/n_z)
    q_v=np.where(gpu_mask, P/die_vol, 0.0)       # measured watts, spread in the die
    Tinf_K=T_case+C2K                            # measured surface pins the coolant side

    T=solve(k,q_v,args.h_top,Tinf_K,L)

    T_junc=T[gpu_mask].max()
    # HBM = device-plane cells that are NOT the die
    hbm_mask=np.zeros_like(gpu_mask); z_dev=int(n_z*0.6)
    hbm_mask[:,:,:z_dev]=True; hbm_mask &= ~gpu_mask
    T_hbm=T[hbm_mask].max()

    print("=== thermal bridge: NVML telemetry -> predicted internal field ===")
    print(f"  MEASURED (NVML) : case {T_case:.0f} C  at {P:.1f} W")
    print(f"  --- solver predicts what the sensor cannot see: ---")
    print(f"  GPU junction    : {T_junc:.1f} C   (+{T_junc-T_case:.1f} K above the sensor)")
    print(f"  HBM (neighbour) : {T_hbm:.1f} C   <- lateral crosstalk from the die")
    print(f"  field range     : {T.min():.1f} .. {T.max():.1f} C over {n_lat}x{n_lat}x{n_z} cells")
    if args.vti:
        write_vti(args.vti,T,L)
        print(f"  wrote {args.vti}  (open in ParaView: hot die, warm HBM flanks)")
    print(f"\n  One sensor reading became a full 3D field. That gap -- surface to")
    print(f"  junction, and die to memory -- is what a thermal-aware diagnostic adds.")

if __name__ == "__main__":
    main()
