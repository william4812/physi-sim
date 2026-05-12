[![physi-sim CI](https://github.com/william4812/physi-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/william4812/physi-sim/actions/workflows/ci.yml)

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

* **Metric:** Max Absolute Difference is strictly bounded at **$5.00 \times 10^{-4}$**.
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

## Technical Stack
* **Engine:** C++ Solver Logic with high-performance Fortran numerical kernels.
* **Analytics:** Automated Python pipeline (PyVista, Matplotlib, Pandas) for headless profiling.
* **CI/CD:** Automated GTest suites ensuring physical conservation laws remain intact during refactoring.

## Getting Started
### Prerequisites
* CMake (>= 3.10)
* G++ (Standard 17)
* GTest (Google Test Framework)

### Build and Test
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ctest --output-on-failure

## Getting Started
### Prerequisites
* CMake (>= 3.10)
* G++ (Standard 17)
* GTest (Google Test Framework)
