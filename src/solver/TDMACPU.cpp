// src/solver/TDMACPU.cpp
#include "solver/TDMACPU.hpp"
#include "solver/fortran_kernels.hpp"

namespace physi_sim::solver {

void TDMACPU::step(core::Grid2D& grid) 
{
    const int nx = grid.get_nx();
    const int ny = grid.get_ny();
    laplace_2d_tdma(grid.data(), nx, ny, &residual_);
    
    history_.push_back(residual_);
}

const std::vector<double>& TDMACPU::get_history() const
{                                                             
    return history_; // Return your existing member 
}


double      TDMACPU::residual() const { return residual_; }
std::string TDMACPU::name()     const { return "TDMACPU"; }

} // namespace physi_sim::solver
