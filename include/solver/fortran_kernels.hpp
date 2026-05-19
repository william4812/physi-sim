// include/solver/fortran_kernels.hpp
// Phase 4: extern "C" bridge to diffusion_kernel.f90 bind(C) subroutines.
#pragma once

extern "C" {
    void laplace_2d_jacobi(const double* T, double* T_new,
                           int nx, int ny, double* res_norm);

    void laplace_2d_tdma(double* T, int nx, int ny, double* res_norm);
}
