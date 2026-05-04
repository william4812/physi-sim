#include "io/TelemetryWriter.hpp"
#include "io/VTKWriter.hpp"
#include "thermal/FortranBackend.hpp"
#include "iostream"

// The Treaty: We still need this to tell the Linker about the Fortran symbol
extern "C" 
{
    void compute_diffusion_f90(const double* T_now, 
                               double* T_next, 
                               int nx, 
                               double r);
}

// Link to the Fortran backend
extern "C" 
{
    void thermal_1d_steady(int n, double dx, double k, 
                           double T_left, double T_right, double* T_in, double* T_out,
                           double& res_max);
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
void FortranBackend::compute_steady(std::vector<double>& T, double k, double T_L, double T_R) 
{
    int n = static_cast<int>(T.size());
    double dx = 1.0 / (n - 1);
    double res = 1.0;
    const double tolerance = 1e-6; // convergence criteria
    int iter = 0;
    std::vector<double> T_old = T;
    std::vector<double> convergence_history;

    std::cout << "\n[Solver] Starting Iterative Steady-State Solve...\n";

    while (res > tolerance && iter < 1000)
    {
        T_old = T;
        // Call the new TDMA-based Fortran kernel
        thermal_1d_steady(n, dx, k, T_L, T_R, T_old.data(), T.data(), res);
        convergence_history.push_back(res);
        ++iter;

        if (iter % 1 == 0)
        {
            std::cout << "Iteration " << iter
                      << "| Relative Error " << res
                      << "\n";
        }
    }

    // Professional Telemetry Integration
    physi_sim::io::TelemetryWriter logger("output");
    if (!logger.save_convergence(convergence_history)) 
    {
        std::cerr << "[Error] Failed to write telemetry data.\n";
    }

    physi_sim::io::VTKWriter vtk_writer;
    vtk_writer.write(T, "output/result_1d.vtk");

    std::cout << "[IO] Steady-state results exported to VTK.\n";
}

} // namespace physi_sim::thermal
