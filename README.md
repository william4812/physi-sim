# LBM-PINN: High-Performance Thermal Simulation

A modular Lattice Boltzmann Method solver with Physics-Informed Neural Network (PINN) integration.

## Key Architecture
- **Language:** Modern C++17/20
- **Design Pattern:** Strategy Pattern via `IComputeBackend`
- **HPC Targets:** AVX-512 (CPU) and CUDA (GPU)
- **Documentation:** Modular LaTeX system with Doxygen API generation.

## Current Progress
- [x] Phase 1, Day 1: Infrastructure & Scaffolding established.
- [x] Modern C++ Interface Guards (RAII, Anti-slicing).
