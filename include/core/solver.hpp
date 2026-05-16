#pragma once
#include <vector>
#include "core/Grid2D.hpp"
#include <cstddef>

namespace physi_sim
{
namespace core
{

class Solver2D 
{
public:
    Solver2D() = default;

    // High-level method that the user (and tests) call
    double solve_laplace_jacobi(Grid2D& grid, 
                         double tolerance = 1e-7, 
                         int max_iter = 5000,
                         std::vector<double>* history = nullptr // Default to null
                             );
    
    double solve_laplace_tdma(Grid2D& grid, 
                         double tolerance = 1e-7, 
                         int max_iter = 5000,
                         std::vector<double>* history = nullptr // Default to null
                             );
    
};

} // namespace core
} // namespace physi_sim
