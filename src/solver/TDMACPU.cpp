// src/solver/TDMACPU.cpp
#include "solver/TDMACPU.hpp"
#include "solver/fortran_kernels.hpp"

namespace physi_sim::solver {

void TDMACPU::step(core::Grid2D& grid) {
    const int nx = grid.get_nx();
    const int ny = grid.get_ny();
    laplace_2d_tdma(grid.data(), nx, ny, &residual_);
}

double      TDMACPU::residual() const { return residual_; }
std::string TDMACPU::name()     const { return "TDMACPU"; }

} // namespace physi_sim::solver
