#include "core/solver.hpp"
#include <vector>
#include <cmath>
#include <iostream>

// Link to the Fortran kernel
extern "C" 
{
    void laplace_2d_jacobi(const double* T, double* T_new, int nx, int ny, double* res_norm);
    void laplace_2d_tdma(const double* T, int nx, int ny, double* res_norm);
}

namespace physi_sim
{
namespace core 
{

double Solver2D::solve_laplace_jacobi(Grid2D& grid, 
        double tolerance, int max_iter,
        std::vector<double>* history) 
{
    int nx = grid.get_nx();
    int ny = grid.get_ny();
    
    // Create a secondary buffer for the "Ping-Pong" update
    std::vector<double> next_data(nx * ny);
    // Initialize next_data with current grid values (to preserve boundaries)
    next_data = grid.get_raw_vector(); 

    double residual = 1.0;
    int iter = 0;

    while (residual > tolerance && iter < max_iter) {
        double current_res = 0.0;
        
        // Call the Fortran Optimized Kernel
        laplace_2d_jacobi(grid.data(), next_data.data(), nx, ny, &current_res);
        
        // Swap data back to the main grid (The "Ping-Pong" swap)
        grid.update_data(next_data);
        
        residual = current_res;

        if (history)
        {
            history->push_back(residual);
        }

        
        if (iter % 100 == 0) {
            std::cout << "[Solver] Iteration " << iter << " | Residual: " << residual << std::endl;
        }
        iter++;
    }

    return residual;
}

double Solver2D::solve_laplace_tdma(Grid2D& grid, 
        double tolerance, int max_iter,
        std::vector<double>* history) 
{
    int nx = grid.get_nx();
    int ny = grid.get_ny();

    double residual = 1.0;
    int iter = 0;

    while (residual > tolerance && iter < max_iter) {
        // The Fortran kernel updates grid.data() directly
        // and returns the Max Relative Error into the residual address
        laplace_2d_tdma(grid.data(), nx, ny, &residual);

        if (history) 
        {
            history->push_back(residual);
        }
        
        if (iter % 100 == 0) 
        {
            std::cout << "[Solver] Iteration " << iter << " | Residual: " << residual << std::endl;
        }
        iter++;
    }

    std::cout << "[Solver] Final TDMA Convergence at Iteration " << iter << " | Final Residual: " << residual << std::endl;
    return residual;
}

} // namespace core
} // namespace physisim
