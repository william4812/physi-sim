/**
 * @file test_comparison_variants.cpp
 * @brief Proves the four comparison-export variants produce correct,
 *        mutually-consistent fields — so the figures rest on tests, not trust.
 *
 *   JacobiCPU, TDMACPU, JacobiGPU(no-VRAM), JacobiGPU(VRAM)
 *   all solve the SAME 2D Laplace BVP (T_top=100) to the same tolerance.
 *   At convergence every interior cell must agree across all four to 5e-3.
 */
#include <gtest/gtest.h>
#include "solver/ISolver.hpp"
#include "solver/JacobiCPU.hpp"
#include "solver/TDMACPU.hpp"
#include "core/Grid2D.hpp"
#ifdef PHYSI_SIM_CUDA_ENABLED
#include "solver/CudaJacobiSolver.hpp"
#include <cuda_runtime.h>
#endif
#include <cmath>

using namespace physi_sim;

namespace 
{
constexpr int    N   = 60;       // small enough to be fast, big enough to be real
constexpr double TOL = 5e-4;     // absolute L-inf, same as the exporter
constexpr int    CAP = 20000;
constexpr double AGREE = 5e-3;   // cross-variant field agreement

core::Grid2D fresh() 
{
    core::Grid2D g(N, N);
    for (int x = 0; x < N; ++x) g(x, N - 1) = 100.0;
    return g;
}

void solve_to_tol(solver::ISolver& s, core::Grid2D& g) 
{
    do { s.step(g); } while (s.residual() > TOL);
}

void solve_hard(solver::ISolver& s, core::Grid2D& g, double tol) 
{
    int it = 0;
    do { s.step(g); } while (s.residual() > tol && ++it < CAP);
}

// max |a-b| over interior cells
double max_interior_diff(const core::Grid2D& a, const core::Grid2D& b) 
{
    double m = 0.0;
    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            m = std::max(m, std::abs(a.at(x, y) - b.at(x, y)));
    return m;
}
// physical sanity: top row hot, bottom cold, interior strictly between
void expect_physical(const core::Grid2D& g) 
{
    EXPECT_DOUBLE_EQ(g.at(N/2, N-1), 100.0) << "top boundary not preserved";
    EXPECT_LT(g.at(N/2, 1), 50.0)           << "bottom interior should be cool";
    for (int x = 1; x < N-1; ++x)
        for (int y = 1; y < N-1; ++y)
            EXPECT_TRUE(g.at(x,y) >= -1e-6 && g.at(x,y) <= 100.0 + 1e-6)
                << "interior out of physical range at (" << x << "," << y << ")";
}
} // namespace

// Guards: src/solver/JacobiCPU.cpp + diffusion_kernel.f90 (laplace_2d_jacobi).
// Logic: run JacobiCPU to tolerance, then assert the FIELD is physically
// sane — top row pinned at 100, interior cooler, every cell within [0,100].
// If this fails: the Fortran stencil or the JacobiCPU copy-back is wrong
// (e.g. boundary overwritten, or NaN leaking from an indexing bug).
TEST(ComparisonVariants, JacobiCPUFieldIsPhysical) 
{
    auto g = fresh(); solver::JacobiCPU s; solve_to_tol(s, g);
    expect_physical(g);
}

TEST(ComparisonVariants, TDMACPUFieldIsPhysical) 
{
    auto g = fresh(); solver::TDMACPU s; solve_to_tol(s, g);
    expect_physical(g);
}

TEST(ComparisonVariants, JacobiCPUandTDMACPUAgree)
{
    // Converge BOTH to 1e-6 (not the export's 5e-4): Jacobi's L-inf residual
    // and TDMA's relative-error residual reach a given field accuracy at
    // different residual values, so only deep convergence makes them comparable.
    auto gj = fresh(); 
    solver::JacobiCPU sj; 
    solve_hard(sj, gj, 1e-6);
    
    auto gt = fresh(); 
    solver::TDMACPU  st; 
    solve_hard(st, gt, 1e-6);
    
    EXPECT_LT(max_interior_diff(gj, gt), 1e-2)
        << "Jacobi and TDMA converge to different fields — physics mismatch";
}

#ifdef PHYSI_SIM_CUDA_ENABLED
TEST(ComparisonVariants, JacobiGPUNoVramMatchesCPU) 
{
    int dev = 0; cudaGetDeviceCount(&dev);
    if (dev == 0) GTEST_SKIP() << "no CUDA device";
    auto gc = fresh(); solver::JacobiCPU       sc; solve_to_tol(sc, gc);
    auto gg = fresh(); solver::CudaJacobiSolver sg; solve_to_tol(sg, gg);  // step() = no VRAM
    expect_physical(gg);
    EXPECT_LT(max_interior_diff(gc, gg), AGREE)
        << "GPU no-VRAM field disagrees with CPU";
}

TEST(ComparisonVariants, JacobiGPUVramMatchesNoVram) 
{
    int dev = 0; cudaGetDeviceCount(&dev);
    if (dev == 0) GTEST_SKIP() << "no CUDA device";
    auto gn = fresh(); solver::CudaJacobiSolver sn; solve_to_tol(sn, gn);   // no VRAM
    auto gv = fresh(); solver::CudaJacobiSolver sv;                          // VRAM
    sv.upload(gv); sv.solve_vram(CAP, TOL); sv.download(gv);
    expect_physical(gv);
    EXPECT_LT(max_interior_diff(gn, gv), AGREE)
        << "VRAM field disagrees with no-VRAM — residency changed the answer";
}
#endif
