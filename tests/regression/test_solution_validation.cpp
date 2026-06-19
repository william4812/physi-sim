#include <gtest/gtest.h>
#include <vector>
#include "core/Grid2D.hpp"
#include "core/solver.hpp"
    
// max |T(i,j) − ¼(N+S+E+W)| over interior cells.
// Independent of the stopping rule: it asks whether the field *solves the PDE*.
double maxLaplaceResidual(const std::vector<double>& T, int nx, int ny) {
    double worst = 0.0;
    for (int j = 1; j < ny - 1; ++j)
        for (int i = 1; i < nx - 1; ++i) {
            const double center    = T[j*nx + i];                       // column-major: [j*nx+i]
            const double neighbors = T[j*nx + (i+1)] + T[j*nx + (i-1)]
                                   + T[(j+1)*nx + i] + T[(j-1)*nx + i];
            worst = std::max(worst, std::abs(center - 0.25 * neighbors));
        }
    return worst;
}

TEST(SolutionValidation, ConvergedFieldSatisfiesDiscreteLaplace) {
    // ARRANGE: standard problem (T_top=100, others=0), JacobiCPU, 50×50.
    //   >>> copy the solver build + run-to-convergence + field accessor
    //       straight from test_comparison_variants.cpp <
    //   For the RED step, set the solver tolerance LOOSE on purpose: 1e-2.
    const int nx = 50;
    const int ny = 50;
    physi_sim::core::Grid2D grid(nx, ny);
    for (int x = 0; x < nx; ++x)
        grid.at(x, ny - 1) = 100.0;          // top edge hot; other edges default to 0

    physi_sim::core::Solver2D solver;
    std::vector<double> history;

    // ACT: run to convergence; read the final field into `T` (size nx*ny).
solver.solve_laplace_jacobi(grid, /*tol=*/1e-6, /*max_it=*/10000, &history);

    // the solve mutates `grid` in place, so the solved field is now in it
    std::vector<double> T = grid.get_raw_vector();
    // (if get_raw_data() returns a raw pointer rather than a vector, use:
    //  std::vector<double> T(grid.get_raw_data(), grid.get_raw_data() + nx*ny); )

    // precondition guard — fail loud, not segfault
    ASSERT_EQ(T.size(), static_cast<std::size_t>(nx) * ny)
        << "field not populated — did the solve run?";

    // ASSERT:
    const double residual = maxLaplaceResidual(T, nx, ny);
    std::cout << "max discrete-Laplace residual = " << residual << "\n";
    EXPECT_LT(residual, 1e-5);
}

TEST(SolutionValidation, Smoke) 
{ 
    SUCCEED();//an explicit "this passed" marker 
}
