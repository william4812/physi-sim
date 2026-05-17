[![physi-sim CI](https://github.com/william4812/physi-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/william4812/physi-sim/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/tests-45%20passing-brightgreen)

# Physi-Sim: High-Performance Thermal Simulation Engine

A C++/Fortran hybrid simulation suite designed for high-fidelity thermal analysis. This project implements advanced numerical schemes—including **LBL-TDMA** (Line-by-Line) and **Jacobi** point-iterative solvers—to demonstrate computational efficiency and rigorous numerical verification in HPC environments.

<p align="center">
  <img src="docs/figures/thermal_verification_composite_2d.png" width="850px">
  <br>
  <b>Figure 1: Steady-State 2D Thermal Field Analysis.</b>
  <br>
  <i>Verification of the 2D Laplacian operator showing stable gradient development.</i>
</p>

## 1. Algorithmic Efficiency
**First Principle:** Iterative convergence rate and spectral radius optimization.

<p align="center">
  <img src="docs/figures/convergence_comparison.png" width="700px">
  <br>
  <b>Figure 2: Numerical Convergence Profile (Jacobi vs. LBL-TDMA).</b>
</p>

* **Metric:** The implicit **LBL-TDMA** solver achieves a $10^{-7}$ residual in ~1,500 iterations, outperforming the **Jacobi** method (~5,500 iterations).
* **HPC Impact:** A **3.6x reduction** in time-to-solution, minimizing computational overhead for large-scale grid simulations.

## 2. Numerical Consistency & Rigor
**First Principle:** Algorithmic cross-validation to bound numerical drift.

<p align="center">
  <img src="docs/figures/2d_consistency_check.png" width="850px">
  <br>
  <b>Figure 3: Field-to-Field Consistency Analysis.</b>
</p>

* **Metric:** Max Absolute Difference is strictly bounded at $5.00 \times 10^{-4}$.
* **Reliability:** Proving the optimized TDMA solver matches the Jacobi baseline ensures numerical integrity for mission-critical applications.

## 3. Fundamental Physical Validation
**First Principle:** 1st-order verification against analytical steady-state solutions.

<p align="center">
  <img src="docs/figures/verification_plot.png" width="600px">
  <br>
  <b>Figure 4: 1D Thermal Diffusion Benchmark.</b>
</p>

* **Accuracy:** Perfect agreement with the analytical $T=x$ profile across a 100-point domain.
* **Verification:** Validates the core Finite Volume discretization before scaling to complex geometries.

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
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```
