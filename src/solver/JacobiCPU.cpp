// src/solver/JacobiCPU.cpp
#include "solver/JacobiCPU.hpp"
#include "solver/fortran_kernels.hpp"
#include <vector>

namespace physi_sim::solver 
{

void JacobiCPU::step(core::Grid2D& grid) 
{
    const int nx = grid.get_nx();
    const int ny = grid.get_ny();

    // Fortran laplace_2d_jacobi needs a separate output array.
    // Grid2D data is row-major: data[j*nx+i] = grid(i,j).
    // Fortran T(nx,ny) column-major accesses same offsets — no transpose needed.
    std::vector<double> T_new(nx * ny);

    laplace_2d_jacobi(grid.data(), T_new.data(), nx, ny, &residual_);

    // Copy result back — Fortran preserves boundary values in T_new
    for (int j = 1; j < ny - 1; ++j)
        for (int i = 1; i < nx - 1; ++i)
            grid(i, j) = T_new[j * nx + i];


    history_.push_back(residual_);
}

double      JacobiCPU::residual() const { return residual_; }
std::string JacobiCPU::name()     const { return "JacobiCPU"; }

} // namespace physi_sim::solver
