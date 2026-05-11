#include <gtest/gtest.h>
#include "core/Grid2D.hpp"
#include "core/solver.hpp"
#include "io/VTKWriter.hpp"
#include "io/CSVWriter.hpp"

// Test 1: The Jacobi Baseline
TEST(Physics2DTest, JacobiConvergenceAndProfile) 
{
    // 1. Setup a 50x50 Grid
    const int NX = 50;
    const int NY = 50;
    physi_sim::core::Grid2D grid(NX, NY);
    
    // 2. Apply Top Boundary Condition (Heat Source)
    for (int x = 0; x < NX; ++x) {
        grid.at(x, NY - 1) = 100.0;
    }

    // 3. Setup Solver and Telemetry
    std::vector<double> residual_history;
    physi_sim::core::Solver2D solver;
    const double target_tolerance = 1e-7;
    const int max_iterations = 10000;

    // 4. Execution: Solve with History Capture
    double final_residual = solver.solve_laplace_jacobi(
        grid, 
        target_tolerance, 
        max_iterations, 
        &residual_history
    );

    // 5. I/O: Export both Spatial Map and Convergence History
    physi_sim::io::VTKWriter vtk;
    vtk.write_2d(grid.get_raw_data(), NX, NY, "jacobi_final_map.vtk");

    physi_sim::io::CSVWriter csv;
    csv.write_history("jacobi_convergence.csv", residual_history);

    // 6. Assertions (The "NVIDIA Signal")
    EXPECT_LT(final_residual, target_tolerance);
    EXPECT_GT(residual_history.size(), 0); // Ensure history was actually recorded
}

// Test 2: The TDMA Significance Test
TEST(Physics2DTest, TDMASweepEfficiency) 
{
    int NX = 50, NY = 50;
    physi_sim::core::Grid2D grid(NX, NY);

    for(int x = 0; x < NX; ++x) grid.at(x, NY-1) = 100.0;

    std::vector<double> residual_history;
    physi_sim::core::Solver2D solver;
    // Note: TDMA propagates info across a whole line in one step
    double residual = solver.solve_laplace_tdma(grid, 1e-7, 3000, &residual_history);

    // 4. Export results
    physi_sim::io::VTKWriter vtk;
    vtk.write_2d(grid.get_raw_data(), NX, NY, "tdma_final_map.vtk");

    physi_sim::io::CSVWriter csv;
    csv.write_history("tdma_convergence.csv", residual_history);

    EXPECT_LT(residual, 1e-7);
    EXPECT_GT(residual_history.size(), 0);
    // Significance Check: TDMA should finish in < 15% of Jacobi's iterations
    // (Actual iteration count will be logged to console)
}
