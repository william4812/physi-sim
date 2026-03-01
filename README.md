# LBM-PINN: High-Performance Thermal Simulation
[![physi-sim CI](https://github.com/william4812/physi-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/william4812/physi-sim/actions/workflows/ci.yml)

## Overview
A high-performance Lattice Boltzmann Method (LBM) solver integrated with Physics-Informed Neural Networks (PINN) for thermal simulation. 

## Engineering "Horizon" (CI/CD)
This project utilizes a **Continuous Integration** pipeline to ensure:
* **Build Integrity:** Parallel matrix builds for `Debug` and `Release` configurations.
* **Physics Validation:** Automated TDD suite verifying conservation laws and data integrity.
* **Automated Deployment:** CD pipeline for generating shippable release artifacts.

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
