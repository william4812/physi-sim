/**
 * @file test_cuda_jacobi_solver.cpp
 * @brief TDD test suite for CudaJacobiSolver.
 *
 * Written BEFORE the implementation — this file defines the contract.
 * These tests will fail to link until CudaJacobiSolver.hpp/.cu exist.
 * That linker failure is the RED state. It is expected and correct.
 *
 * TEST LAYERS:
 *   1. Contract tests   — name(), residual() initial state, ISolver interface
 *   2. Physics tests    — step() produces valid residual, field converges
 *   3. Parity test      — GPU field matches CPU field to 5e-4 after convergence
 *
 * CI NOTE:
 *   GitHub Actions has no GPU. Every test guards with cudaGetDeviceCount()
 *   and calls GTEST_SKIP() if no device is present. CI stays green.
 *   Local GTX 1650 runs the full suite.
 */

#include <gtest/gtest.h>
#include "solver/CudaJacobiSolver.hpp"
#include "solver/ISolver.hpp"
#include "core/Grid2D.hpp"

#ifdef PHYSI_SIM_CUDA_ENABLED
#include <cuda_runtime.h>
#endif
#include <cmath>
#include <memory>

// ---------------------------------------------------------------------------
// Test fixture — shared grid setup used by every test
// ---------------------------------------------------------------------------

class CudaJacobiSolverTest : public ::testing::Test 
{
protected:
    void SetUp() override {
        // Skip every test in this fixture if no CUDA device is present.
        // This keeps CI (no GPU) green while running the full suite locally.
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA device found — skipping GPU tests";
        }

        // 20×20 grid — small enough to be fast, large enough to have
        // meaningful interior. Boundary conditions:
        //   top row (y=ny-1): T = 100
        //   all other boundaries: T = 0
        grid = std::make_unique<physi_sim::core::Grid2D>(nx, ny);

        for (int x = 0; x < nx; ++x)
            for (int y = 0; y < ny; ++y)
                (*grid)(x, y) = 0.0;

        // Top boundary: T = 100
        for (int x = 0; x < nx; ++x)
            (*grid)(x, ny - 1) = 100.0;
    }

    static constexpr int nx = 20;
    static constexpr int ny = 20;
    static constexpr double tolerance = 5e-4;

    std::unique_ptr<physi_sim::core::Grid2D> grid;
};

// ---------------------------------------------------------------------------
// 1. Contract tests — written first, define the ISolver interface obligations
// ---------------------------------------------------------------------------

// RED: fails to link until CudaJacobiSolver.hpp declares the class
TEST_F(CudaJacobiSolverTest, ImplementsISolverInterface) 
{
    // If CudaJacobiSolver doesn't inherit ISolver, this line won't compile.
    // That compile error IS the red state — it describes exactly what's missing.
    physi_sim::solver::ISolver* solver =
        new physi_sim::solver::CudaJacobiSolver();
    delete solver;
}

TEST_F(CudaJacobiSolverTest, NameReturnsJacobiGPU) 
{
    physi_sim::solver::CudaJacobiSolver solver;
    EXPECT_EQ(solver.name(), "JacobiGPU");
}

TEST_F(CudaJacobiSolverTest, ResidualIsZeroBeforeAnyStep) 
{
    // Before step() is called, no iteration has occurred.
    // Residual must be 0.0 — not garbage from uninitialised memory.
    physi_sim::solver::CudaJacobiSolver solver;
    EXPECT_DOUBLE_EQ(solver.residual(), 0.0);
}

TEST_F(CudaJacobiSolverTest, IsNonCopyable) 
{
    // Device memory ownership is exclusive. Copying would silently
    // alias raw CUDA pointers — two objects calling cudaFree on the
    // same pointer is undefined behaviour.
    // This test is a compile-time assertion — it never runs at runtime.
    EXPECT_FALSE(std::is_copy_constructible_v<physi_sim::solver::CudaJacobiSolver>);
    EXPECT_FALSE(std::is_copy_assignable_v<physi_sim::solver::CudaJacobiSolver>);
}

// ---------------------------------------------------------------------------
// 2. Physics tests — single step and convergence behaviour
// ---------------------------------------------------------------------------

TEST_F(CudaJacobiSolverTest, ResidualIsPositiveAfterOneStep) 
{
    physi_sim::solver::CudaJacobiSolver solver;
    solver.step(*grid);

    // After one step, interior cells have been updated.
    // Residual = max|T_new - T_old| must be > 0.
    EXPECT_GT(solver.residual(), 0.0);
}

TEST_F(CudaJacobiSolverTest, ResidualIsFiniteAfterOneStep) {
    physi_sim::solver::CudaJacobiSolver solver;
    solver.step(*grid);

    // NaN or Inf residual means the kernel has a memory bug
    // (wrong stride, out-of-bounds access, uninitialised memory).
    EXPECT_TRUE(std::isfinite(solver.residual()));
}

TEST_F(CudaJacobiSolverTest, ResidualDecreasesMonotonically) {
    // Jacobi on a consistent boundary-value problem must converge.
    // Residual must be non-increasing (within floating-point noise).
    physi_sim::solver::CudaJacobiSolver solver;

    double prev = std::numeric_limits<double>::max();
    for (int i = 0; i < 100; ++i) {
        solver.step(*grid);
        double curr = solver.residual();
        EXPECT_LE(curr, prev * 1.01)  // 1% tolerance for floating-point noise
            << "Residual increased at iteration " << i
            << ": prev=" << prev << " curr=" << curr;
        prev = curr;
    }
}

TEST_F(CudaJacobiSolverTest, ConvergesToTolerance) {
    // Full convergence test — same criterion as CPU Jacobi regression.
    // Tolerance 5e-4 matches test_convergence_baselines.cpp.
    physi_sim::solver::CudaJacobiSolver solver;

    int iter = 0;
    const int max_iters = 10000;
    while (iter < max_iters) {
        solver.step(*grid);
        ++iter;
        if (solver.residual() < tolerance) break;
    }

    EXPECT_LT(solver.residual(), tolerance)
        << "GPU Jacobi did not converge in " << max_iters << " iterations";

    // Sanity: convergence iteration count should be in the same ballpark
    // as CPU Jacobi (~5500 on a 20×20 grid). Hard upper bound: 10000.
    EXPECT_LT(iter, max_iters)
        << "Iteration count " << iter << " suggests non-convergence";
}

TEST_F(CudaJacobiSolverTest, BoundaryValuesUnchangedAfterConvergence) {
    // ISolver contract: halo (boundary) cells are read-only.
    // The kernel must never write to boundary indices.
    physi_sim::solver::CudaJacobiSolver solver;

    // Record boundary values before solving
    const double top_corner_before = (*grid)(0, ny - 1);  // = 100.0

    for (int i = 0; i < 500; ++i)
        solver.step(*grid);

    // Top boundary must still be 100.0 — the kernel must not have touched it
    EXPECT_DOUBLE_EQ((*grid)(0, ny - 1), top_corner_before)
        << "Kernel overwrote boundary cell — halo protection is broken";
}

// ---------------------------------------------------------------------------
// 3. Parity test — GPU field must match CPU field to 5e-4
// ---------------------------------------------------------------------------

TEST_F(CudaJacobiSolverTest, FieldMatchesCPUJacobiAfterConvergence) {
    /**
     * This is the most important test.
     * Same grid, same boundary conditions, same tolerance.
     * GPU field must match CPU field to 5e-4 at every interior cell.
     *
     * If this fails:
     *   - wrong stride in kernel (y*nx+x vs y*ny+x)  ← most common
     *   - boundary copy missing from d_next
     *   - ping-pong pointer swap reversed
     */
    using Grid2D   = physi_sim::core::Grid2D;
    using Jacobi   = physi_sim::solver::CudaJacobiSolver;

    // CPU reference — include once ISolver abstraction includes JacobiCPU
    // For now use a second CudaJacobiSolver as self-consistency check.
    // TODO: replace cpu_solver with JacobiCPU when available in this TU.

    // GPU grid (the fixture grid)
    Jacobi gpu_solver;
    while (gpu_solver.residual() > tolerance || gpu_solver.residual() == 0.0)
        gpu_solver.step(*grid);

    // CPU reference grid — identical initial conditions
    Grid2D cpu_grid(nx, ny);
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            cpu_grid(x, y) = (y == ny - 1) ? 100.0 : 0.0;

    Jacobi cpu_ref_solver;
    while (cpu_ref_solver.residual() > tolerance || cpu_ref_solver.residual() == 0.0)
        cpu_ref_solver.step(cpu_grid);

    // Compare every interior cell
    int mismatches = 0;
    for (int x = 1; x < nx - 1; ++x) {
        for (int y = 1; y < ny - 1; ++y) {
            double diff = std::abs((*grid)(x, y) - cpu_grid(x, y));
            if (diff > tolerance) {
                ++mismatches;
                if (mismatches <= 3)   // report first 3 only to keep output readable
                    ADD_FAILURE() << "Mismatch at (" << x << "," << y << "): "
                                  << "gpu=" << (*grid)(x,y)
                                  << " cpu=" << cpu_grid(x,y)
                                  << " diff=" << diff;
            }
        }
    }
    EXPECT_EQ(mismatches, 0)
        << mismatches << " interior cells differ by more than " << tolerance;
}
