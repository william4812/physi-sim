// src/solver/JacobiCPU.cpp
#include "solver/JacobiCPU.hpp"
#include <cmath>
#include <algorithm>

namespace physi_sim::solver {

void JacobiCPU::step(core::Grid2D& grid) {
    const int nx = grid.get_nx();
    const int ny = grid.get_ny();
    residual_ = 0.0;
    auto old_data = grid.get_raw_vector();
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            double T_new = 0.25 * (
                old_data[(j-1)*nx + i] +
                old_data[(j+1)*nx + i] +
                old_data[j*nx + (i-1)] +
                old_data[j*nx + (i+1)]
            );
            residual_ = std::max(residual_, std::abs(T_new - grid(i, j)));
            grid(i, j) = T_new;
        }
    }
}

double      JacobiCPU::residual() const { return residual_; }
std::string JacobiCPU::name()     const { return "JacobiCPU"; }

} // namespace physi_sim::solver
