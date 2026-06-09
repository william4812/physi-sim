// tests/unit/test_cuda_tdma_solver.cpp
//
// TDD spec for CudaTDMASolver. The CPU TDMA result is GROUND TRUTH; the GPU
// solver must reproduce it. With the stub kernel this FAILS — that is Rung 0.
//
// TODO(you): wire the marked lines to your real Grid2D / TDMACPU API.
//            Mirror tests/unit/test_comparison_variants.cpp — it already does
//            cross-variant parity, so copy its problem setup and run loop.

#include <gtest/gtest.h>
#include "solver/ISolver.hpp"
#include "solver/CudaTDMASolver.hpp"
#include "solver/TDMACPU.hpp"        // TODO(you): your CPU TDMA header/class name
#include "core/Grid2D.hpp"
#include <cuda_runtime.h>

using namespace physi_sim;

namespace {

bool hasCudaDevice() 
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Same boundary setup your other parity tests use (e.g. hot top edge).
core::Grid2D makeProblem(int n) 
{
    core::Grid2D g(n, n);
    for (int x = 0; x < n; ++x) g(x, n - 1) = 100.0;   // T_top = 100, exactly like fresh()
    return g;
}

void runToConvergence(solver::ISolver& s, core::Grid2D& g,
                      int max_iter, double tol) 
{
    for (int k = 0; k < max_iter; ++k) {
        s.step(g);
        if (s.residual() < tol) break;   // TODO(you): match your stop criterion
    }
}

} // namespace

// ── RUNG 2 target: full-grid parity with the CPU reference ──────────────────
TEST(CudaTDMASolverTest, FieldMatchesCPUTDMA) {
    if (!hasCudaDevice()) GTEST_SKIP() << "no CUDA device";

    const int    n          = 32;
    const int    max_iter   = 20000;
    const double tol        = 1e-7;
    const double parity_tol = 5e-4;   // same slack your Jacobi parity test uses

    // Ground truth: CPU TDMA
    core::Grid2D ref = makeProblem(n);
    solver::TDMACPU cpu;
    runToConvergence(cpu, ref, max_iter, tol);

    // Under test: GPU TDMA, VRAM-resident path
    core::Grid2D gpu = makeProblem(n);
    solver::CudaTDMASolver dev;
    dev.upload(gpu);
    dev.solve_vram(max_iter, tol);
    dev.download(gpu);

    for (int j = 1; j < n - 1; ++j)
        for (int i = 1; i < n - 1; ++i)
            EXPECT_NEAR(gpu.at(i, j), ref.at(i, j), parity_tol)   // TODO(you): your accessor
                << "mismatch at (" << i << "," << j << ")";
}


// ── RUNG 1 (write this one FIRST): one tridiagonal system in isolation ──────
// Before the 2D test passes, prove your Thomas on ONE known system so you can
// tell "algorithm wrong" apart from "thread/memory mapping wrong".
//
namespace physi_sim::solver 
{
void tdma_solve_single(const double*, const double*, const double*,
                       const double*, double*, int);
}

TEST(CudaTDMASolverTest, SolvesSingleKnownSystem) 
{
     if (!hasCudaDevice()) GTEST_SKIP() << "no CUDA device";
     // Build a small system (n=4) with a hand-computed answer, launch the
     // kernel with a SINGLE line, assert each x_i ~ expected within 1e-9.

     // inside the test:
    double a[4]={0,-1,-1,-1}, b[4]={2,2,2,2}, c[4]={-1,-1,-1,0}, d[4]={0,0,0,5}, x[4];
    solver::tdma_solve_single(a,b,c,d,x,4);
    const double expect[4]={1,2,3,4};
    for (int i=0;i<4;++i) EXPECT_NEAR(x[i], expect[i], 1e-9);
}

// ------------------------------------------------------------------
// 2. Physics tests — single step and convergence behaviour
// ------------------------------------------------------------------
                              
TEST(CudaTDMASolverTest, ResidualIsPositiveAfterOneStep)         
{
    core::Grid2D grid = makeProblem(32);     // hot top edge → one sweep moves the interior
    physi_sim::solver::CudaTDMASolver solver;
    solver.step(grid);       
                              
    // After one step, interior cells have been updated.
    // Residual = max|T_new - T_old| must be > 0.
    EXPECT_GT(solver.residual(), 0.0);
} 
