[![physi-sim CI](https://github.com/william4812/physi-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/william4812/physi-sim/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/tests-97%20passing-brightgreen)
![Language](https://img.shields.io/badge/language-C%2B%2B20%20%7C%20Fortran%2090%20%7C%20CUDA%2013.2%20%7C%20Python%203-blue)

# physi-sim — HPC Thermal Solver

C++20 / Fortran 90 / CUDA 13.2 hybrid engine for 2D steady-state heat transfer.
GTX 1650 (sm_75, 16 SMs) · TDD · CI/CD · CMakePresets

Implements a modular `ISolver` interface with FSM-managed lifecycle, concurrent
dispatch, and a Fortran physics kernel pipeline. GPU acceleration via CUDA Jacobi
and CUDA TDMA solvers, each with residual history and a `ProfilingHarness`. Kept
**resident in VRAM** across the solve loop, the GPU Jacobi solver runs **~11×
faster than the CPU** at 500×500 — while the GPU TDMA solver, profiled honestly,
**loses** to the CPU because its line-serial structure starves the device. The
result is a controlled study of *when* GPU acceleration helps and when it does
not — see §6.

---

## Table of contents

1. [Architecture](#1-architecture)
2. [Solver lifecycle — Finite State Machine](#2-solver-lifecycle--finite-state-machine)
3. [Physics verification](#3-physics-verification)
4. [Algorithm comparison — Jacobi vs TDMA](#4-algorithm-comparison--jacobi-vs-tdma)
5. [GPU profiling — the PCIe wall](#5-gpu-profiling--the-pcie-wall)
6. [VRAM residency, parallelism, and the limits of acceleration](#6-vram-residency-parallelism-and-the-limits-of-acceleration)
7. [Test suite — 97 passing](#7-test-suite--97-passing)
8. [Build and reproduce results](#8-build-and-reproduce-results)
9. [Roadmap and physics vision](#9-roadmap-and-physics-vision)

---

## 1. Architecture

Four strictly separated layers. C++ owns software design. Fortran owns numerical
physics. CUDA owns GPU compute. Python owns analytics.

```
┌──────────────────────────────────────────────────────────────────┐
│  C++ layer — software architecture                               │
│                                                                  │
│  ISolver (interface: step · residual · name · history)           │
│    ├── JacobiCPU          — Jacobi 5-point stencil (Fortran)    │
│    ├── TDMACPU            — Thomas algorithm line sweep          │
│    ├── CudaJacobiSolver   — CUDA sm_75, 16×16 blocks  ✅        │
│    └── CudaTDMASolver     — CUDA batched line-Jacobi  ✅        │
│                                                                  │
│  SolverFactory    — registry, HardwareBackend, parse CLI        │
│  SolverFSM        — atomic<SolverState>, throws on bad trans.   │
│  ProfilingHarness — wall_ms · iterations · fsm_state → CSV      │
│  ConcurrentSolverRunner — std::async dispatch                   │
└──────────────┬───────────────────────────────────────────────────┘
               │  extern "C" / Fortran bind(C)
               │  zero-overhead ABI, column-major layout
┌──────────────▼───────────────────────────────────────────────────┐
│  Fortran 90 layer — physics kernels (diffusion_kernel.f90)       │
│                                                                  │
│  laplace_2d_jacobi  — 5-point stencil, L∞ residual              │
│  laplace_2d_tdma    — row sweep, calls solve_tdma               │
│  solve_tdma         — PURE Thomas algorithm (thread-safe)        │
└──────────────┬───────────────────────────────────────────────────┘
               │  __global__ kernels / Thrust reduction
┌──────────────▼───────────────────────────────────────────────────┐
│  CUDA layer — GPU compute (src/cuda/)                            │
│                                                                  │
│  jacobi_kernel        — 16×16 thread blocks, sm_75              │
│  tdma_sweep_kernel    — one thread per row, serial Thomas        │
│  copy_boundary_kernel — halo preservation                        │
│  AbsDiff + thrust::max_element — L∞ residual reduction          │
│  VRAM-resident buffers — field stays on device across the loop  │
│  m_history            — per-step residual → convergence CSV     │
└──────────────┬───────────────────────────────────────────────────┘
               │  ProfilingHarness.writeCSV() + history()
┌──────────────▼───────────────────────────────────────────────────┐
│  Python analytics pipeline (python/)                             │
│                                                                  │
│  scripts/plot_convergence.py  — residual vs iteration           │
│  scripts/plot_walltime.py     — wall-time scaling + residency   │
│  scripts/plot_thermal_map.py  — VTK field heatmaps (no pyvista) │
│  scripts/make_figures.py      — one-run orchestrator (composes) │
└──────────────────────────────────────────────────────────────────┘
```

The C++ source mirrors these seams as independent CMake components:
`src/thermal` (Fortran bridge), `src/io` (VTK + config), `src/core` (FSM,
profiling, runner), `src/solver` (ISolver + factory), `src/cuda` (GPU). Each is
its own static library, linked into `physi_sim` and `physi_tests`.

---

## 2. Solver lifecycle — Finite State Machine

Every solver run is governed by `SolverFSM`. State stored as
`std::atomic<SolverState>` — safe to poll from a monitoring thread without a mutex.
Invalid transitions throw `std::logic_error` immediately.

```
IDLE ──prepare()──► READY ──start()──► RUNNING
                                           │
                               ┌───────────┴──────────┐
                           finish(true)         finish(false)
                               │                      │
                           CONVERGED               FAILED
                               └────────reset()────────┘
                                           │
                                         IDLE
```

| Transition | Pre-state | Post-state | On violation |
|------------|-----------|------------|--------------|
| `prepare()` | IDLE | READY | `std::logic_error` |
| `start()` | READY | RUNNING | `std::logic_error` |
| `finish(true)` | RUNNING | CONVERGED | `std::logic_error` |
| `finish(false)` | RUNNING | FAILED | `std::logic_error` |
| `reset()` | **any** | IDLE | — never throws |

`ProfilingHarness::run()` drives the full cycle: `prepare()` → `start()` →
iteration loop → `finish(converged)` → `reset()`. The terminal state is
serialised as `fsm_state` in every CSV row — correctness-tagged, not just timed.

The FSM also drives the benchmark driver in `main.cpp`:

```
BenchmarkState::INIT → RUNNING → WRITING → DONE
```

---

## 3. Physics verification

**Problem:** 2D steady-state Laplace equation, T_top = 100, all other boundaries = 0.

<p align="center">
  <img src="docs/figures/thermal_field.png" width="520px"><br>
  <b>Figure 1 — Converged 500×500 temperature field (JacobiCPU).</b>
  Hot top edge (T = 100) diffusing into a cold interior — the expected
  steady-state harmonic profile.
</p>

All six solver variants — JacobiCPU, TDMACPU, and the GPU Jacobi and GPU TDMA
solvers each in per-step and VRAM-resident form — converge to the same field.
Side by side they are visually indistinguishable:

<p align="center">
  <img src="docs/figures/thermal_compare_500.png" width="1100px"><br>
  <b>Figure 2 — Field parity across all six variants (500×500).</b>
  Differences sit at the level of floating-point reordering across parallel
  threads — orders of magnitude below the convergence tolerance.
</p>

Parity is enforced in CI by the comparison-variant tests
(`FieldMatchesCPUJacobiAfterConvergence`, tol = 5×10⁻⁴): a backend that drifts
from the CPU Jacobi reference fails the build. The GPU TDMA solver added in
Phase 5 reaches the same field, locked by `CudaTDMASolverTest.FieldMatchesCPUTDMA`.

---

## 4. Algorithm comparison — Jacobi vs TDMA

**Question:** under a common stopping criterion, how much faster does the
implicit line-by-line TDMA solver converge than explicit Jacobi?

<p align="center">
  <img src="docs/figures/convergence_100x100.png" width="720px"><br>
  <b>Figure 3 — Jacobi vs LBL-TDMA convergence.</b>
  100×100 grid · relative increment residual ‖Tᵏ−Tᵏ⁻¹‖∞ / ‖T¹−T⁰‖∞ · tol = 1×10⁻⁷.
</p>

| Solver | Iterations to tol | Ratio |
|--------|-------------------|-------|
| JacobiCPU | 17,887 | 1× baseline |
| TDMACPU | 4,842 | **3.69× fewer** |

TDMA reaches tolerance in **3.69× fewer iterations** because each line sweep
propagates information across an entire row implicitly, while Jacobi relaxes one
neighbour at a time.

**Read this as per-iteration efficiency, not wall-clock speed.** Each TDMA
iteration costs ~7× more than a Jacobi sweep (a full Thomas forward+backward pass
per row), so in wall-clock terms Jacobi is actually faster at these grids — see
§5. The increment residual measures how fast the iteration *settles*; the true
solution error is the equation residual ‖b − A·T‖∞, which `ProfilingHarness`
records separately.

The 3.69× iteration ratio is pinned by a CI regression assertion in
`test_convergence_baselines.cpp`: if TDMA ever regresses toward Jacobi's
iteration count, the build fails with the counts in the message.

---

## 5. GPU profiling — the PCIe wall

**Correctness first.** JacobiGPU traces the *same* convergence path as
JacobiCPU — identical iteration counts at every grid size (17,887 at 100×100) —
because it is the same stencil and tolerance. The port is numerically faithful;
this is asserted by the `CudaJacobiSolver` unit tests.

**Wall time.** The naive GPU path copies the field host↔device on every `step()`.
That per-iteration PCIe traffic, not the kernel, dominates:

<p align="center">
  <img src="docs/figures/wall_time.png" width="720px"><br>
  <b>Figure 4 — Wall time to convergence vs grid size (log-log, tol = 1×10⁻⁷).</b>
</p>

| Grid | JacobiCPU | JacobiGPU (per-step) | TDMACPU |
|------|-----------|----------------------|---------|
| 50×50 | 19 ms | 663 ms | 38 ms |
| 100×100 | 277 ms | 2,351 ms | 532 ms |
| 200×200 | 4,272 ms | 13,091 ms | 8,648 ms |

<sub>GTX 1650, single run; absolute timings vary run-to-run, the ordering does not.</sub>

The kernel itself is trivially cheap; each iteration's cost is the round trip
over a ~16 GB/s bus. Wall time climbs steeply — roughly O(N⁴), since Jacobi takes
O(N²) iterations (5,064 → 17,887 → 61,074) each doing O(N²) work — yet the GPU
curve runs a fixed factor *above* the CPU in lockstep on the log-log plot: that
gap is the per-iteration PCIe round trip, not arithmetic. Phase 4 removes it.

---

## 6. VRAM residency, parallelism, and the limits of acceleration

The PCIe wall in §5 is structural, not numerical: copying the field host↔device
every iteration costs more than the kernel. The fix is to own the device
buffer's lifetime — upload **once**, iterate entirely in VRAM, download
**once** — so PCIe is paid twice per *solve* instead of twice per *iteration*.
The controlled comparison below runs all six variants on the identical 500×500
problem to the same tolerance, isolating two independent axes: which algorithm
runs, and where its data lives.

<p align="center">
  <img src="docs/figures/wall_time_residency.png" width="820px"><br>
  <b>Figure 5 — Wall time by variant, 500×500 controlled comparison (log scale).</b><br>
  Hue = algorithm (blue Jacobi, orange TDMA); shade = residency (light CPU,
  medium GPU-per-iteration, dark GPU-resident). Read each color block left to right.
</p>

| Variant (500×500) | Wall time | vs CPU baseline |
|-------------------|-----------|-----------------|
| JacobiCPU | 22,499 ms | 1× (Jacobi baseline) |
| JacobiGPU — per-step (NoVram) | 35,744 ms | 1.6× slower than CPU Jacobi |
| **JacobiGPU — VRAM-resident** | **1,995 ms** | **11.3× faster than CPU Jacobi** |
| TDMACPU | 68,957 ms | 1× (TDMA baseline) |
| TDMAGPU — per-step (NoVram) | 158,708 ms | 2.3× slower than CPU TDMA |
| TDMAGPU — VRAM-resident | 102,297 ms | 1.5× slower than CPU TDMA |

<sub>GTX 1650, single run; absolute timings vary run-to-run (~±20%, thermal
throttling), the ordering does not.</sub>

**Jacobi (blue).** Holding algorithm and hardware fixed and changing only
residency, the GPU goes from 35.7 s (copying every iteration — *slower* than the
22.5 s CPU) to **2.0 s resident**: a **17.9× collapse from data movement alone**,
and **11.3× under the CPU**. Same kernel, same silicon; only the data path changed.

**TDMA (orange).** Residency helps here too (159 s → 102 s) but never drops the
bar below the CPU's 69 s. TDMA's line-by-line Thomas sweep is *sequential within
each row* — its only parallelism is across rows, ~500 independent threads on a
device built for thousands — so the GPU runs at a few percent of capacity no
matter how the data is staged.

The fastest of all six is the *least* sophisticated algorithm (Jacobi) on the GPU
with a disciplined data path; the *most* numerically efficient one (TDMA, which
converges in 3.69× fewer iterations — §4) is the **slowest**, because its serial
structure starves the hardware. This is the central HPC lesson the project was
built to demonstrate: **acceleration follows the parallel structure of the
algorithm and the discipline of the data path — not the choice of chip.** The
fields are bit-for-bit identical across all variants (Figure 2), so every timing
difference is pure hardware/algorithm fit, free of any accuracy cost.

Closing the GPU-TDMA gap is not a coding problem but an algorithmic one: the
Thomas recurrence has an O(N) critical-path *depth* per line, so the device's
parallelism has nothing to chew on. The structural fix is parallel cyclic
reduction — O(log N) depth at the cost of more total work — tracked as roadmap
**Phase 5b**.

---

## 7. Test suite — 97 passing

97 tests across three independent layers; the full suite runs in ~11 s.

```
tests/
├── unit/
│   ├── test_matrix_layout.cpp           Grid2D memory stride
│   ├── test_vtk_writer.cpp              VTKWriter header contract
│   ├── test_config_loader.cpp           ConfigLoader + SimulationParams
│   ├── test_backend_dispatch.cpp        backend dispatch pattern
│   ├── test_solver_factory.cpp          SolverFactory + ISolver contract
│   ├── test_solver_fsm.cpp              SolverFSM lifecycle + bad transitions
│   ├── test_comparison_variants.cpp     cross-variant field parity
│   ├── test_residual_history_csv.cpp    per-step history → CSV
│   ├── test_residual_and_convergence.cpp  residual math + stopping
│   ├── test_cuda_jacobi_solver.cpp      CudaJacobiSolver (GPU; CUDA build only)
│   └── test_cuda_tdma_solver.cpp        CudaTDMASolver (GPU; CUDA build only)
├── integration/
│   ├── test_fortran_interop.cpp         C++ ↔ Fortran ABI bridge
│   ├── test_profiling_harness.cpp       ProfilingHarness + normalized residual
│   └── test_concurrent_runner.cpp       std::async dispatch + speedup proof
└── regression/
    └── test_convergence_baselines.cpp   Jacobi + TDMA convergence baselines
```

| Layer | A failure here means |
|-------|----------------------|
| unit | a class contract broke |
| integration | a component seam broke |
| regression | the physics drifted |

The CUDA test suites (`Cuda*SolverTest.*`) call `GTEST_SKIP()` when no GPU is
present, so CI stays green on CPU-only runners while still compiling the CUDA
path.

---

## 8. Build and reproduce results

### Prerequisites

```
CMake >= 3.21   C++20   gfortran   CUDA 13.2 (optional)
python3 + pip   matplotlib  pandas  numpy
```

### Build (two CMakePresets)

```bash
# Optimized + hardened — daily use, benchmarks, CI
cmake --preset hpc-release
cmake --build --preset hpc-release
ctest --preset hpc-release

# Bounds-checked debug — run before every PR
# (-O0 -g · gfortran -fcheck=all -fbacktrace · libstdc++ _GLIBCXX_ASSERTIONS)
cmake --preset hpc-debug
cmake --build --preset hpc-debug
ctest --preset hpc-debug
```

### Run the benchmark and regenerate every figure

```bash
cd build

# Grid sweep (50/100/200) + the 500×500 six-variant comparison, writing
# profiling CSVs, convergence CSVs, cmp_*.vtk and cmp_timing.csv
./physi_sim

# Convergence + wall-time scaling + the 500×500 residency bar. make_figures reads
# the sweep grids; the residency bar auto-reads the six variants from cmp_timing.csv.
python3 ../python/scripts/make_figures.py --data . --out ../docs/figures --grid 100 --tol 1e-7

# Field + six-panel parity at the comparison grid (500). The comparison runs at 500,
# so these come straight from cmp_*_500.vtk rather than through make_figures.
python3 ../python/scripts/plot_thermal_map.py cmp_cpu_jacobi_500.vtk \
    --out ../docs/figures/thermal_field.png
python3 ../python/scripts/plot_thermal_map.py \
    cmp_cpu_jacobi_500.vtk:"Jacobi CPU" cmp_cpu_tdma_500.vtk:"TDMA CPU" \
    cmp_gpu_jacobi_novram_500.vtk:"Jacobi GPU NoVram" cmp_gpu_jacobi_vram_500.vtk:"Jacobi GPU Vram" \
    cmp_gpu_tdma_novram_500.vtk:"TDMA GPU NoVram" cmp_gpu_tdma_vram_500.vtk:"TDMA GPU Vram" \
    --out ../docs/figures/thermal_compare_500.png
```

`make_figures.py` skips any figure whose inputs are missing, so a partial
benchmark still produces whatever it can. Grid sizes, tolerance, GPU toggle and
comparison grid live in `benchmark_config.json` — edit and rerun `./physi_sim`
without recompiling.

> Always benchmark from the **`hpc-release`** build. Debug (`-O0`) timings are
> meaningless for the wall-time figures.

---

## 9. Roadmap and physics vision

| Phase | Branch | Description | Status |
|-------|--------|-------------|--------|
| 0 | `feat/isolver-abstraction` | ISolver, SolverFactory, ProfilingHarness | ✅ |
| 1 | `feat/solver-fsm` | SolverFSM — thread-safe, throws on bad transitions | ✅ |
| 2 | `feat/cuda-jacobi` | CudaJacobiSolver — GPU tests, benchmarks | ✅ |
| 3 | `feat/benchmark-driver` | CMakePresets, JSON config, FSM driver, figures | ✅ |
| 4 | `feat/phase2-vram-resident` | VRAM-resident buffers — GPU beats CPU (§6) | ✅ |
| 5a | `feat/cuda-tdma` | CudaTDMASolver — batched per-line Thomas (line-Jacobi), profiled vs CPU | ✅ |
| 5b | `feat/cuda-tdma` | Parallel cyclic reduction (PCR) — intra-line parallelism, O(log N) depth | 🔲 |
| 6 | `feat/shared-memory-tiling` | Jacobi shared-memory halo exchange | 🔲 |
| 7 | `feat/3d-physics` | 3D ADI conduction, Navier–Stokes FVM, P1 radiation | 🔲 |
| 8 | `feat/multiscale-bridge` | MesoBoltzmannSolver, Kn-based ScaleManager | 🔲 |

### The long game: a modular Conjugate Heat Transfer framework

Phases 7–8 grow physi-sim from a single-equation thermal solver into a modular
**Conjugate Heat Transfer (CHT)** framework — conduction, convection and
radiation solved together and coupled at fluid–solid interfaces, all behind the
same `ISolver` contract.

**Three transport modes (macro scale).**

$$\rho C_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + \dot{q} \quad\text{(3-D conduction, implicit ADI/TDMA)}$$

$$\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) = -\nabla p + \nabla \cdot \boldsymbol{\tau} + \rho \mathbf{g} \quad\text{(Navier–Stokes, FVM)}$$

$$\frac{dI(\mathbf{r}, \mathbf{s})}{ds} = \kappa I_b - (\kappa + \sigma_s) I + \frac{\sigma_s}{4\pi}\int_{4\pi} I(\mathbf{r}, \mathbf{s}')\,\Phi(\mathbf{s}', \mathbf{s})\,d\Omega' \quad\text{(RTE, P1 / DOM)}$$

coupled at interfaces by temperature and flux continuity
$T_s = T_f,\; -k_s \,\partial_n T_s = -k_f \,\partial_n T_f$.

**Multi-scale by Knudsen number** ($Kn = \lambda / L$). A `ScaleManager` selects
the governing physics per region as the continuum hypothesis breaks down:

| Scale | Length | Dominant physics |
|-------|--------|------------------|
| Macro | > 100 µm | Continuum transport (Navier–Stokes + Fourier), $Kn < 0.001$ |
| Meso | 1–100 µm | Slip flow / transition, NS + Maxwell-slip BCs, $0.001 < Kn < 0.1$ |
| Nano | 1–1000 nm | Sub-continuum ballistic, Boltzmann Transport Equation |
| Pico | < 1 nm | Quantum / atomic, Schrödinger / ab-initio MD |

**Validated by a TDD bridge.** The safe way across the macro→meso boundary is an
*overlap* where both models are valid. A micro-channel heat-sink case tuned to
$Kn \approx 0.05$ — continuum solver with slip + temperature-jump BCs vs a
Boltzmann/DSMC solver — must produce the same temperature profile. A single
Google Test, `MultiScaleValidationTest.VerifyMacroMesoConsistency`, asserts
profile agreement to 0.01%, giving a mathematical bridge of confidence before
each new physics regime is trusted.
