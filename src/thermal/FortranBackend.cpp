#include "thermal/FortranBackend.hpp"

// The Treaty: We still need this to tell the Linker about the Fortran symbol
extern "C" {
    void compute_diffusion_f90(const double* T_now, 
                               double* T_next, 
                               int nx, 
                               double r);
}

// Link to the Fortran backend
extern "C" {
    void thermal_1d_steady(int n, double dx, double k, double T_left, double T_right, double* T_out);
}

namespace physi_sim::thermal {

void FortranBackend::compute(std::vector<double>& data, double alpha, double dt) {
    // 1st Principle: std::vector stores data contiguously in memory.
    // .data() gives us the raw pointer the Fortran ABI requires.
    
    // For a 1D diffusion, we usually need a 'next' buffer. 
    // In a real HPC solver, you'd manage this to avoid allocations in the loop.
    std::vector<double> next_step = data; 
    
    int nx = static_cast<int>(data.size());
    double dx = 1.0 / (nx - 1);
    double r = (alpha * dt) / (dx * dx);// The diffusion stability factor

    // Call the Fortran kernel using the raw memory bridge
    compute_diffusion_f90(data.data(), next_step.data(), nx, r);

    // Swap the results back into the original vector
    data.swap(next_step);
}

// New Implicit/Steady-State Solver (Today's Work)
void FortranBackend::compute_steady(std::vector<double>& data, double k, double T_L, double T_R) 
{
    int n = static_cast<int>(data.size());
    double dx = 1.0 / (n - 1);

    // Call the new TDMA-based Fortran kernel
    thermal_1d_steady(n, dx, k, T_L, T_R, data.data());
}

} // namespace physi_sim::thermal
