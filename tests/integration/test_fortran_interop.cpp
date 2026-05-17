// tests/integration/test_fortran_interop.cpp
//
// Layer  : integration
// Seam   : C++ <-> Fortran ABI bridge
// Owns   : ABICompatibility · FortranBackend physics verification
// Headers: thermal/FortranBackend.hpp
// Sources: src/fortran/diffusion_kernel.f90
//
// These are integration tests because they cross the language boundary.
// A failure here means the bind(C) linkage or a Fortran kernel broke —
// not a C++ class contract failure.
//
// Split from: tests/unit/test_solver.cpp (lines 80-162)
// LBM tests kept in: tests/unit/test_backend_dispatch.cpp

#include <gtest/gtest.h>
#include "thermal/FortranBackend.hpp"

// ── ABI handshake ─────────────────────────────────────────────────────────────
// check_abi_integrity() doubles its input and returns via out-param.
// 0.125 × 2 = 0.25 is exactly representable in IEEE 754 binary64.
// EXPECT_DOUBLE_EQ (not EXPECT_NEAR) is intentional — if the ABI is
// correct there must be zero rounding error on this value.

extern "C" 
{
    void check_abi_integrity(double val_in, double* val_out);
}

TEST(ABICompatibility, FortranDoubleMatchesCppDouble)
{
    double input  = 0.125;
    double output = 0.0;

    check_abi_integrity(input, &output);

    EXPECT_DOUBLE_EQ(output, 0.25);
}

// ── FortranBackend: explicit diffusion step ───────────────────────────────────
// First-principle: a single heat pulse at the centre of a 3-point grid
// must decrease at the centre and leave Dirichlet boundaries unchanged.

TEST(FortranBackendTest, ConvergenceTest)
{
    using namespace physi_sim::thermal;

    FortranBackend solver;

    std::vector<double> grid = {0.0, 1.0, 0.0};

    double alpha = 0.01;
    double dt    = 1.0;

    solver.compute(grid, alpha, dt);

    EXPECT_LT(grid[1], 1.0);
    EXPECT_EQ(grid[0], 0.0);
    EXPECT_EQ(grid[2], 0.0);
    EXPECT_NEAR(grid[0], grid[2], 1e-10);  // physical symmetry
}

// ── FortranBackend: steady-state TDMA ─────────────────────────────────────────
// First-principle: steady-state 1D conduction with Dirichlet BCs and no
// source term produces a perfectly linear temperature profile.
// Analytical solution: T(i) = T_L + i * (T_R - T_L) / (n - 1)

TEST(FortranBackendTest, SteadyStateLinearProfileTest)
{
    using namespace physi_sim::thermal;

    FortranBackend solver;

    const int    n   = 11;
    const double T_L = 100.0;
    const double T_R = 0.0;
    const double k   = 1.0;

    std::vector<double> grid(n, 0.0);
    solver.compute_steady(grid, k, T_L, T_R);

    const double slope = (T_R - T_L) / (n - 1);
    for (int i = 0; i < n; ++i) {
        const double expected_T = T_L + i * slope;
        EXPECT_NEAR(grid[i], expected_T, 1e-9)
            << "Physical violation at node " << i
            << ": Non-linear profile detected.";
    }

    EXPECT_DOUBLE_EQ(grid[0],   T_L);
    EXPECT_DOUBLE_EQ(grid[n-1], T_R);
}
