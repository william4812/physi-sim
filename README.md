[![physi-sim CI](https://github.com/william4812/physi-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/william4812/physi-sim/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/tests-45%20passing-brightgreen)
![Language](https://img.shields.io/badge/language-C%2B%2B17%20%7C%20Fortran%2090%20%7C%20Python%203-blue)

# Physi-Sim: Multi-Scale Conjugate Heat Transfer Engine

A C++17 / Fortran 90 / Python hybrid simulation engine for high-fidelity thermal analysis across macro, meso, and nano scales. Implements **conjugate heat transfer (CHT)** — coupling solid conduction, fluid convection, and radiant transport — through a modular `ISolver` interface with FSM-managed lifecycle, concurrent dispatch, and a Fortran physics kernel pipeline.

Designed to demonstrate HPC and embedded systems engineering depth: interface abstraction, deterministic state machine lifecycle, thread-safe concurrent dispatch, and a validated multi-scale physics bridge from continuum Navier-Stokes down to Boltzmann transport.

---

## Table of contents

1. [Architecture](#1-architecture)
2. [Solver lifecycle — Finite State Machine](#2-solver-lifecycle--finite-state-machine)
3. [Physics verification results](#3-physics-verification-results)
4. [Multi-scale roadmap](#4-multi-scale-roadmap)
5. [Test suite — 45 / 45 passing](#5-test-suite--45--45-passing)
6. [Language pipeline](#6-language-pipeline)
7. [Getting started](#7-getting-started)

---

## 1. Architecture

Three strictly separated layers. C++ owns software design. Fortran owns numerical physics. Python owns analytics. No layer reaches into another's responsibility.

```
┌──────────────────────────────────────────────────────────────────┐
│  C++ layer — software architecture                               │
│                                                                  │
│  ISolver (interface)                                             │
│    ├── JacobiCPU          — Jacobi point-iterative stencil      │
│    ├── TDMACPU            — line-by-line Thomas algorithm        │
│    ├── CudaThermalSolver  — GPU backend (Phase 5, planned)       │
│    └── MesoBoltzmannSolver— BTE / DSMC (Phase 7, planned)       │
│                                                                  │
│  SolverFactory    — registry, HardwareBackend enum, parse CLI   │
│  SolverFSM        — atomic<SolverState>, throws on bad trans.   │
│  ProfilingHarness — wall_ms · iterations · fsm_state → CSV      │
│  ConcurrentSolverRunner — std::async dispatch (Phase 3)         │
│  ScaleManager     — Kn-based sub-domain delegation (Phase 7)    │
└──────────────┬───────────────────────────────────────────────────┘
               │  extern "C"  /  Fortran bind(C, name="…")
               │  zero-overhead ABI bridge, column-major layout
┌──────────────▼───────────────────────────────────────────────────┐
│  Fortran 90 layer — physics kernels (diffusion_kernel.f90)       │
│                                                                  │
│  laplace_2d_jacobi  — 5-point stencil, L∞ residual              │
│  laplace_2d_tdma    — row sweep, calls solve_tdma                │
│  solve_tdma         — PURE Thomas algorithm (thread-safe)        │
│  thermal_1d_steady  — 1D steady CHT verification                 │
│  [planned] ns_fvm · adi_3d · boltzmann_bte                       │
└──────────────┬───────────────────────────────────────────────────┘
               │  ProfilingHarness.writeCSV()
               │  wall_time_ms · fsm_state · solver · grid_nx …
┌──────────────▼───────────────────────────────────────────────────┐
│  Python layer — analytics pipeline                               │
│                                                                  │
│  plot_profiling.py     — CSV → grouped bar chart (CPU vs GPU)   │
│  verify_2d_consistency.py — field cross-validation, Cartesian   │
│  PyVista               — VTK post-processing, 3D field render   │
└──────────────────────────────────────────────────────────────────┘
```

### Class hierarchy

```
ISolver  (pure virtual: step · residual · name)
├── JacobiCPU           calls laplace_2d_jacobi  (Fortran)
├── TDMACPU             calls laplace_2d_tdma     (Fortran)
├── CudaThermalSolver   -DPHYSI_CUDA=ON  [Phase 5]
└── MesoBoltzmannSolver BTE / DSMC        [Phase 7]

SolverFactory::create(SolverType, HardwareBackend) → unique_ptr<ISolver>
SolverFactory::parseSolverType(string)  → SolverType   (case-insensitive)
SolverFactory::parseBackend(string)     → HardwareBackend

ProfilingHarness(unique_ptr<ISolver>)
  .run(Grid2D, max_iters, tolerance)  → ProfilingRecord
  .writeCSV(path)                     → solver,backend,grid_nx,grid_ny,
                                        iterations,residual,wall_ms,
                                        converged,fsm_state

SolverFSM   — value type, atomic<SolverState>, non-copyable
ConcurrentSolverRunner — std::async, one ProfilingRecord per thread [Phase 3]
ScaleManager           — Kn per block → solver delegation [Phase 7]
```

---

## 2. Solver lifecycle — Finite State Machine

Every solver run is governed by `SolverFSM`, a lightweight thread-safe value type. State is stored as `std::atomic<SolverState>` — `ConcurrentSolverRunner` can safely poll `state()` from a monitoring thread without a mutex. Invalid transitions throw `std::logic_error` immediately with the current state and event name in the message. No silent failures.

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

**Integration with ProfilingHarness:** `run()` calls `prepare()` → `start()` → iteration loop → `finish(converged)` → `reset()`. The terminal state is serialised as `fsm_state` in the CSV — every profiling row is correctness-tagged, not just timed.

**GTest coverage:** 22 tests — all 5 states, all valid transitions, 9 invalid-transition throw cases, error message content (names current state + event), concurrent `state()` read safety (ThreadSanitizer-clean), and three-cycle reuse after `reset()`.

---

## 3. Physics verification results

### 3.1 Algorithmic efficiency — Jacobi vs LBL-TDMA

**First principle:** iterative convergence rate and spectral radius optimisation.

<p align="center">
  <img src="docs/figures/convergence_comparison.png" width="700px">
  <br>
  <b>Figure 1: Residual convergence — Jacobi vs LBL-TDMA.</b>
</p>

The implicit LBL-TDMA solver achieves a 10⁻⁷ residual in ~1,500 iterations, versus ~5,500 for Jacobi — a **3.6× reduction** in time-to-solution. The implicit ADI scheme is unconditionally stable (no Fourier number restriction on Δt), enabling large time steps for transient problems.

### 3.2 Numerical consistency — field cross-validation

**First principle:** algorithmic cross-validation to bound numerical drift.

<p align="center">
  <img src="docs/figures/2d_consistency_check.png" width="850px">
  <br>
  <b>Figure 2: Field-to-field consistency analysis (Jacobi vs TDMA, 2D steady-state).</b>
</p>

Max absolute difference strictly bounded at **5.00 × 10⁻⁴**. Both solvers produce physically identical fields — TDMA's 3.6× speed advantage carries no accuracy cost.

### 3.3 Fundamental physical validation — analytical benchmark

**First principle:** first-order verification against an analytical steady-state solution.

<p align="center">
  <img src="docs/figures/verification_plot.png" width="600px">
  <br>
  <b>Figure 3: 1D thermal diffusion — numerical vs analytical T = x profile.</b>
</p>

Perfect agreement with the analytical T = x profile across a 100-point domain. Validates the core Finite Volume discretisation and Fortran `bind(C)` ABI bridge before scaling to complex geometries.

### 3.4 Steady-state 2D thermal field

<p align="center">
  <img src="docs/figures/thermal_verification_composite_2d.png" width="850px">
  <br>
  <b>Figure 4: Steady-state 2D thermal field — stable gradient, Laplacian operator verified.</b>
</p>

Visualised with `origin='lower'` (Cartesian convention) following coordinate-system verification commit. Confirms the 2D Laplacian discretisation and Python analytics pipeline produce physically correct spatial orientation.

---

## 4. Multi-scale roadmap

As the spatial domain shrinks, the continuum hypothesis breaks down. The framework switches governing physics based on the Knudsen number **Kn = λ / L** (mean free path / characteristic length).

### 4.1 Scale transition table

| Scale | Dimension | Kn range | Governing physics | Governing equation | Solver class | Status |
|-------|-----------|----------|-------------------|--------------------|--------------|--------|
| Macro | > 100 μm | < 0.001 | Continuum NS + Fourier | ρCₚ∂T/∂t = ∇·(k∇T) | `MacroContinuumSolver` | 🔲 Ph 6 |
| Meso | 1–100 μm | 0.001–0.1 | NS + Maxwell slip BCs | u_s = [(2−σ_v)/σ_v]λ ∂u/∂n | `MacroContinuumSolver` (slip) | 🔲 Ph 7 |
| Nano | 1–1000 nm | 0.1–10 | Boltzmann Transport | ∂f/∂t + v·∇f = Q(f) | `MesoBoltzmannSolver` | 🔲 Ph 7 |
| Pico | < 1 nm | > 10 | Schrödinger / MD | Hψ = Eψ | `PicoQuantumSolver` | 🔲 future |

### 4.2 Conjugate heat transfer coupling

The macro-scale framework couples three transport modes:

```
ρCₚ ∂T/∂t = ∇·(k∇T) + q̇                        (solid conduction, ADI/TDMA)

∂ρ/∂t + ∇·(ρu) = 0                               (continuity)
∂(ρu)/∂t + ∇·(ρu⊗u) = −∇p + ∇·τ + ρg           (momentum)
∂(ρE)/∂t + ∇·(u(ρE+p)) = ∇·(k_f∇T + u·τ)        (energy)

dI(r,s)/ds = κI_b − (κ+σ_s)I + (σ_s/4π)∫I Φ dΩ  (RTE, P1 approximation)
```

CHT interface conditions at the fluid–solid boundary:

```
T_solid|Γ = T_fluid|Γ                   (temperature continuity)
−k_s ∂T_s/∂n = −k_f ∂T_f/∂n           (flux continuity)
```

### 4.3 TDD bridge — Macro ↔ Meso at Kn = 0.05

At **Kn ≈ 0.05** (micro-channel, L = 50 μm), both the continuum solver with Maxwell slip BCs and the Boltzmann solver are valid. They must produce identical temperature profiles to ε = 1×10⁻⁴ (0.01%) before the simulation descends into the non-continuum nano-scale regime.

```cpp
// tests/test_multiscale_kn005.cpp  [Phase 7]
TEST_F(MultiScaleValidationTest, MacroMesoConsistency) {
    macro_solver.EnableSlipBoundaryConditions(true);
    auto macro_r = macro_solver.SolveToSteadyState();
    auto meso_r  = meso_solver.SolveToSteadyState();

    for (size_t i = 0; i < macro_r.temperature_profile.size(); ++i)
        EXPECT_NEAR(macro_r.temperature_profile[i],
                    meso_r.temperature_profile[i],
                    macro_r.temperature_profile[i] * 1e-4)
            << "Divergence at spatial index: " << i;
}
```

This test is the mathematical proof-of-confidence. If it passes, both physics models agree from first principles — the bridge is safe to cross.

### 4.4 Implementation phases

| Phase | Branch | Description | Status |
|-------|--------|-------------|--------|
| 0 | `feat/isolver-abstraction` | ISolver + SolverFactory + ProfilingHarness | ✅ merged |
| 1 | `feat/solver-fsm` | SolverFSM — 22 GTests, thread-safe | ✅ merged |
| 2 | `feat/fsm-harness-integration` | FSM wired into ProfilingHarness, `fsm_state` in CSV | 🔲 next |
| 3 | `refactor/test-layer-structure` | unit/integration/regression layers, 45 tests | ✅ merged |
| 4 | `feat/concurrent-solver-runner` | `std::async` dispatch, concurrent wall-time proof | 🔲 |
| 5 | `feat/fortran-bridge` | `extern "C"` bridge replaces C++ stencil copies | 🔲 |
| 6 | `feat/cuda-stub` | `CudaThermalSolver` via `-DPHYSI_CUDA=ON` | 🔲 |
| 7 | `feat/3d-physics` | 3D ADI conduction, NS FVM, P1 radiation | 🔲 |
| 8 | `feat/multiscale-bridge` | `MesoBoltzmannSolver`, `ScaleManager`, Kn=0.05 TDD | 🔲 |
| 9 | `docs/report` | CPU vs GPU profiling table, LaTeX writeup | 🔲 |
---

## Test Architecture

45 tests organised into three independent layers. Each layer has a distinct
failure meaning and can be run in isolation.

```
tests/
├── unit/                          one class in isolation, no physics
│   ├── test_matrix_layout.cpp     core::Grid2D memory stride
│   ├── test_vtk_writer.cpp        io::VTKWriter header contract
│   ├── test_config_loader.cpp     io::ConfigLoader + SimulationParams
│   ├── test_backend_dispatch.cpp  Backend dispatch pattern (pre-ISolver)
│   ├── test_solver_factory.cpp    SolverFactory + ISolver contract
│   └── test_solver_fsm.cpp        SolverFSM lifecycle (22 tests)
├── integration/                   real components at their seams
│   └── test_fortran_interop.cpp   C++ ↔ Fortran ABI bridge
└── regression/                    physics tolerances vs known-good baselines
    └── test_convergence_baselines.cpp  Jacobi + TDMA convergence
```

| Layer | Tests | A failure here means |
|-------|-------|----------------------|
| unit | 40 | A class contract broke |
| integration | 3 | A language-boundary seam broke |
| regression | 2 | Physics drifted |

Run each layer independently:

```bash
cd build
ctest -L unit        --output-on-failure   # < 1 second
ctest -L integration --output-on-failure
ctest -L regression  --output-on-failure
ctest                --output-on-failure   # full suite
```

The TDMA 3.6× speedup claim is enforced by a regression assertion — not just
a comment. If TDMA ever regresses toward Jacobi speed, CI fails with the
iteration count in the failure message.

---

## Technical Stack

* **Engine:** C++17 solver logic with high-performance Fortran 90 numerical kernels.
* **Analytics:** Automated Python pipeline (PyVista, Matplotlib, Pandas) for headless profiling.
* **CI/CD:** 45 GTests across unit / integration / regression layers — physical conservation
  laws and convergence rates are verified on every push.

## Getting Started

### Prerequisites

* CMake (>= 3.14)
* G++ (C++17)
* gfortran
* GTest (fetched automatically via CMake FetchContent)

### Build and Test

```bash
# Required
cmake >= 3.15
g++ >= 9          # C++17
gfortran >= 9     # Fortran 90 kernels
libgtest-dev      # or fetched automatically via CMake FetchContent

# Optional
python3 + pip     # analytics pipeline
nvcc              # CUDA toolkit >= 11.0, for Phase 5
```

### Build and test

```bash
# Standard CPU build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```
