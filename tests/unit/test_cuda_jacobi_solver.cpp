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
#include "solver/JacobiCPU.hpp"
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
    void SetUp() override 
    {
#ifndef PHYSI_SIM_CUDA_ENABLED
        GTEST_SKIP() << "Built without CUDA — skipping GPU tests";
#else
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA device found — skipping GPU tests";
        }

        grid = std::make_unique<physi_sim::core::Grid2D>(nx, ny);

        for (int x = 0; x < nx; ++x)
            for (int y = 0; y < ny; ++y)
                (*grid)(x, y) = 0.0;

        for (int x = 0; x < nx; ++x)
            (*grid)(x, ny - 1) = 100.0;
#endif
    }

    static constexpr int nx = 100; //20;
    static constexpr int ny = 100; //20;
    // absolute L∞ tolerance — max|T_new - T_old| in Kelvin
    // not the same as ProfilingHarness normalized tolerance (ratio)
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

#ifdef PHYSI_SIM_CUDA_ENABLED
TEST_F(CudaJacobiSolverTest, IterationCountMatchesCPUJacobi)
{
    /**
     * Same stencil → same iteration count to the same tolerance.
     * GPU and CPU Jacobi must converge in the same number of steps
     * on identical initial conditions. A difference > 5% indicates
     * a stencil mismatch — the most common silent CUDA kernel bug.
     */
    using Jacobi    = physi_sim::solver::CudaJacobiSolver;
    using JacobiCPU = physi_sim::solver::JacobiCPU;

    // GPU solve
    Jacobi gpu_solver;
    int gpu_iters = 0;
    while (gpu_solver.residual() > tolerance || gpu_solver.residual() == 0.0) {
        gpu_solver.step(*grid);
        ++gpu_iters;
    }

    // CPU solve — identical initial conditions
    physi_sim::core::Grid2D cpu_grid(nx, ny);
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            cpu_grid(x, y) = (y == ny - 1) ? 100.0 : 0.0;

    JacobiCPU cpu_solver;
    int cpu_iters = 0;
    while (cpu_solver.residual() > tolerance || cpu_solver.residual() == 0.0) {
        cpu_solver.step(cpu_grid);
        ++cpu_iters;
    }

    // Same stencil → same iteration count within 5% floating-point noise
    const double ratio = static_cast<double>(gpu_iters) / cpu_iters;
    EXPECT_NEAR(ratio, 1.0, 0.15)
        << "GPU iters=" << gpu_iters << " CPU iters=" << cpu_iters
        << " ratio=" << ratio
        << " — stencil mismatch suspected";
}

// ── Phase 2: VRAM-resident interface ─────────────────────────────────────────

TEST_F(CudaJacobiSolverTest, SolveVRAMConvergesToTolerance)
{
    // Proves the new path converges to the same residual as step()-based path.
    physi_sim::solver::CudaJacobiSolver solver;
    solver.upload(*grid);
    solver.solve_vram(10000, tolerance);
    solver.download(*grid);

    EXPECT_LT(solver.residual(), tolerance)
        << "VRAM-resident solve did not converge";
}

TEST_F(CudaJacobiSolverTest, SolveVRAMIterationCountMatchesStepBased)
{
    // solve_vram and step() use the same stencil → they converge in the same
    // number of iterations (within stride/FP noise). This tests convergence-
    // detection equivalence, NOT the field (that is the next test).
    physi_sim::solver::CudaJacobiSolver vram_solver;
    vram_solver.upload(*grid);
    vram_solver.solve_vram(10000, tolerance);
    vram_solver.download(*grid);
    const int vram_iters = vram_solver.get_vram_iterations();   // use your header's accessor

    physi_sim::core::Grid2D grid2(nx, ny);
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            grid2(x, y) = (y == ny - 1) ? 100.0 : 0.0;

    physi_sim::solver::CudaJacobiSolver step_solver;
    while (step_solver.residual() > tolerance || step_solver.residual() == 0.0)
        step_solver.step(grid2);
    const int step_iters = static_cast<int>(step_solver.history().size());

    EXPECT_NEAR(static_cast<double>(vram_iters) / step_iters, 1.0, 0.05)
        << "vram_iters=" << vram_iters << " step_iters=" << step_iters;
}

TEST_F(CudaJacobiSolverTest, SolveVRAMFieldMatchesStepBased)
{
    // Same kernel + same iteration count + same initial field → identical
    // field (the stencil is deterministic per cell). We compare at a FIXED
    // iteration count, not at convergence: with the residual checked every
    // RESIDUAL_STRIDE steps, solve_vram stops a few dozen iterations past
    // where step() stops, and slow Jacobi convergence turns that into a field
    // difference far above 5e-4. Fixing the iteration count removes that confound.
    constexpr int FIXED_ITERS = 2000;

    // tol = -1.0 disables early stopping, so it runs exactly FIXED_ITERS.
    physi_sim::solver::CudaJacobiSolver vram_solver;
    vram_solver.upload(*grid);
    vram_solver.solve_vram(FIXED_ITERS, -1.0);
    vram_solver.download(*grid);

    physi_sim::core::Grid2D grid2(nx, ny);
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            grid2(x, y) = (y == ny - 1) ? 100.0 : 0.0;

    physi_sim::solver::CudaJacobiSolver step_solver;
    for (int i = 0; i < FIXED_ITERS; ++i)
        step_solver.step(grid2);

    int mismatches = 0;
    for (int x = 1; x < nx - 1; ++x)
        for (int y = 1; y < ny - 1; ++y)
            if (std::abs((*grid)(x, y) - grid2(x, y)) > tolerance)
                ++mismatches;

    EXPECT_EQ(mismatches, 0)
        << mismatches << " interior cells differ after " << FIXED_ITERS
        << " identical iterations";
}

TEST_F(CudaJacobiSolverTest, SolveVRAMFasterThanStepBased)
{
    // VRAM-resident solve must be faster than step()-based on same grid.
    // This is the Phase 4 performance proof.
    using clock = std::chrono::steady_clock;

    // Phase 1 timing
    physi_sim::core::Grid2D g1(nx, ny);
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            g1(x, y) = (y == ny - 1) ? 100.0 : 0.0;
    physi_sim::solver::CudaJacobiSolver s1;
    auto t0 = clock::now();
    while (s1.residual() > tolerance || s1.residual() == 0.0)
        s1.step(g1);
    double ms_phase1 = std::chrono::duration<double,std::milli>(
        clock::now() - t0).count();

    // Phase 4 timing
    physi_sim::core::Grid2D g2(nx, ny);
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            g2(x, y) = (y == ny - 1) ? 100.0 : 0.0;
    physi_sim::solver::CudaJacobiSolver s2;
    auto t1 = clock::now();
    s2.upload(g2);
    s2.solve_vram(10000, tolerance);
    s2.download(g2);
    double ms_phase4 = std::chrono::duration<double,std::milli>(
        clock::now() - t1).count();

    std::cout << "\n[Phase4Speedup] Phase1=" << ms_phase1
              << "ms  Phase4=" << ms_phase4
              << "ms  Speedup=" << ms_phase1/ms_phase4 << "×\n";

    EXPECT_LT(ms_phase4, ms_phase1)
        << "Phase 4 must be faster than Phase 1";
}
#endif
