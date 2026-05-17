// tests/regression/test_convergence_baselines.cpp
//
// Layer  : regression
// Owns   : 2D physics convergence — residual tolerance + TDMA efficiency
// Headers: core/Grid2D.hpp · core/solver.hpp · io/VTKWriter.hpp · io/CSVWriter.hpp
//
// A failure here means physics drifted, not a software bug.
// From: tests/unit/test_physics.cpp

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

    // 6. Assertions
    EXPECT_LT(final_residual, target_tolerance);
    EXPECT_GT(residual_history.size(), 0u);
}

// Test 2: The TDMA Significance Test
TEST(Physics2DTest, TDMASweepEfficiency) 
{
    int NX = 50, NY = 50;
    physi_sim::core::Grid2D grid(NX, NY);

    for (int x = 0; x < NX; ++x) grid.at(x, NY-1) = 100.0;

    std::vector<double> residual_history;
    physi_sim::core::Solver2D solver;
    double residual = solver.solve_laplace_tdma(grid, 1e-7, 3000, &residual_history);

    physi_sim::io::VTKWriter vtk;
    vtk.write_2d(grid.get_raw_data(), NX, NY, "tdma_final_map.vtk");

    physi_sim::io::CSVWriter csv;
    csv.write_history("tdma_convergence.csv", residual_history);

    EXPECT_LT(residual, 1e-7);
    EXPECT_GT(residual_history.size(), 0u);

    // Significance Check: TDMA must converge in < 30% of Jacobi's iterations.
    // Jacobi baseline ~5500 iters. Threshold = 5500 * 0.30 = 1650.
    // Was a comment in test_physics.cpp — now a verified CI assertion.
    const size_t jacobi_baseline = 5500;
    EXPECT_LT(residual_history.size(),
              static_cast<size_t>(jacobi_baseline * 0.30))
        << "TDMA efficiency regression: took " << residual_history.size()
        << " iterations (threshold: " << static_cast<size_t>(jacobi_baseline * 0.30) << ")";
}
