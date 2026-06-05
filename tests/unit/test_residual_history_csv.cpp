/**
 * @file test_residual_history_csv.cpp
 * @brief Guards the residual-history CSVs that feed the README §5 CPU-vs-GPU
 *        profiling plots. Two things must hold for those plots to be honest:
 *
 *   (1) Every solver's history is a valid sequence of the SAME quantity —
 *       the absolute L-inf norm of the iterate update:
 *           r_k = || T^k - T^{k-1} ||_inf = max_ij | T^k(i,j) - T^{k-1}(i,j) |
 *       computed identically by laplace_2d_jacobi (diffusion_kernel.f90:121),
 *       laplace_2d_tdma (diffusion_kernel.f90:161), and CudaJacobiSolver
 *       (CudaJacobiSolver.cu:280). Same definition => the curves are
 *       directly comparable on one axis.
 *
 *   (2) The history round-trips through CSVWriter::write_history into the
 *       "Iteration,Residual" schema without losing, reordering, or
 *       corrupting samples.
 *
 * VRAM x-axis contract: solve_vram() records the residual only every
 * RESIDUAL_STRIDE iterations, so its history is SPARSE — the CSV row index is
 * a SAMPLE number, and the true iteration is index * RESIDUAL_STRIDE. The
 * GPUStepHistoryIsDensePerIteration vs VramHistoryIsStrided pair locks that,
 * so the plot x-axis is built correctly (scale the VRAM x by the stride).
 *
 * Accessor: uses get_history() (returns std::vector<double> by value) — it is
 * defined on all three solvers (JacobiCPU.cpp, TDMACPU.cpp,
 * CudaJacobiSolver.cu), so this test needs NO header changes to compile.
 */
#include <gtest/gtest.h>
#include "solver/ISolver.hpp"
#include "solver/JacobiCPU.hpp"
#include "solver/TDMACPU.hpp"
#include "core/Grid2D.hpp"
#include "io/CSVWriter.hpp"
#ifdef PHYSI_SIM_CUDA_ENABLED
#include "solver/CudaJacobiSolver.hpp"
#include <cuda_runtime.h>
#endif
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace physi_sim;

namespace {
constexpr int    N   = 60;       // small enough to be fast, real enough to converge
constexpr double TOL = 5e-4;     // absolute L-inf increment (same as the exporter)
constexpr int    CAP = 20000;    // hard ceiling so a broken solver fails, not hangs

core::Grid2D fresh() {
    core::Grid2D g(N, N);
    for (int x = 0; x < N; ++x) g(x, N - 1) = 100.0;   // T_top = 100, rest 0
    return g;
}

void solve_to_tol(solver::ISolver& s, core::Grid2D& g) {
    int it = 0;
    do { s.step(g); } while (s.residual() > TOL && ++it < CAP);
}

// Assert a residual-history vector is a physically valid convergence sequence.
void expect_valid_history(const std::vector<double>& h, const char* who) {
    ASSERT_FALSE(h.empty()) << who << ": history is empty — nothing was logged";
    for (size_t i = 0; i < h.size(); ++i) {
        EXPECT_TRUE(std::isfinite(h[i]))
            << who << ": non-finite residual at sample " << i
            << " — kernel memory bug (bad stride / OOB / uninit)";
        EXPECT_GE(h[i], 0.0)
            << who << ": negative residual at sample " << i
            << " — an L-inf norm cannot be < 0";
    }
    EXPECT_GT(h.front(), 0.0)
        << who << ": first residual is 0 — solver did no work on the first step";
    // Non-increasing within 1% slack (matches CudaJacobiSolverTest convention;
    // tolerates floating-point noise without masking real divergence).
    for (size_t i = 1; i < h.size(); ++i)
        EXPECT_LE(h[i], h[i - 1] * 1.01)
            << who << ": residual increased at sample " << i
            << " (" << h[i - 1] << " -> " << h[i] << ") — iteration is diverging";
    EXPECT_LT(h.back(), h.front())
        << who << ": last residual not below first — no convergence at all";
}

// Parse an "Iteration,Residual" CSV back into header + the two columns.
struct Csv { std::string header; std::vector<long> iter; std::vector<double> res; };
Csv read_csv(const std::string& path) {
    std::ifstream f(path);
    Csv d;
    std::getline(f, d.header);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const auto comma = line.find(',');
        d.iter.push_back(std::stol(line.substr(0, comma)));
        d.res .push_back(std::stod(line.substr(comma + 1)));
    }
    return d;
}
} // namespace

// ── Per-solver history validity ───────────────────────────────────────────────

// Guards: JacobiCPU.cpp + laplace_2d_jacobi — absolute L-inf of the update.
TEST(ResidualHistoryCSV, JacobiCPUHistoryIsValidSequence) {
    auto g = fresh(); solver::JacobiCPU s; solve_to_tol(s, g);
    expect_valid_history(s.get_history(), "JacobiCPU");
    EXPECT_LT(s.get_history().back(), TOL) << "JacobiCPU did not reach tolerance";
}

// Guards: TDMACPU.cpp + laplace_2d_tdma — absolute L-inf of the update,
// the SAME definition as Jacobi (confirmed by reading the kernel).
TEST(ResidualHistoryCSV, TDMACPUHistoryIsValidSequence) {
    auto g = fresh(); solver::TDMACPU s; solve_to_tol(s, g);
    expect_valid_history(s.get_history(), "TDMACPU");
    EXPECT_LT(s.get_history().back(), TOL) << "TDMACPU did not reach tolerance";
}

// ── CSV round-trip — the artifact the README actually plots ────────────────────

// Guards: CSVWriter::write_history. A real solver history -> CSV -> parse back
// must preserve count, order, the 0-based Iteration column, and the values.
TEST(ResidualHistoryCSV, HistoryRoundTripsThroughCSV) {
    auto g = fresh(); solver::JacobiCPU s; solve_to_tol(s, g);
    const std::vector<double> h = s.get_history();

    const std::string path = "/tmp/physi_resid_roundtrip.csv";
    io::CSVWriter().write_history(path, h);

    const Csv back = read_csv(path);
    EXPECT_EQ(back.header, "Iteration,Residual")
        << "schema drift — the README plot and the other profiling CSVs all "
           "assume this exact header";
    ASSERT_EQ(back.res.size(), h.size()) << "CSV row count != history length";
    for (size_t i = 0; i < h.size(); ++i) {
        EXPECT_EQ(back.iter[i], static_cast<long>(i))
            << "Iteration column must be 0-based and contiguous (row " << i << ")";
        EXPECT_NEAR(back.res[i], h[i], 1e-9 + 1e-5 * std::abs(h[i]))
            << "residual value corrupted at row " << i
            << " (write/read precision mismatch)";
    }
    std::remove(path.c_str());
}

#ifdef PHYSI_SIM_CUDA_ENABLED
// Guards: CudaJacobiSolver.cu step() path. step() logs one residual per call —
// DENSE history, one sample per iteration, directly comparable to the CPU
// curves. (Same residual definition: tests 47/74/75 already prove the field
// and iteration count match CPU Jacobi.)
TEST(ResidualHistoryCSV, GPUStepHistoryIsDensePerIteration) {
    int dev = 0; cudaGetDeviceCount(&dev);
    if (dev == 0) GTEST_SKIP() << "no CUDA device";
    auto g = fresh(); solver::CudaJacobiSolver s; solve_to_tol(s, g);
    expect_valid_history(s.get_history(), "JacobiGPU(step)");
    EXPECT_LT(s.get_history().back(), TOL) << "JacobiGPU(step) did not reach tolerance";
}

// Guards: CudaJacobiSolver.cu solve_vram() path. solve_vram logs only every
// RESIDUAL_STRIDE iterations => SPARSE history. This test LOCKS that contract:
// the CSV row index is a SAMPLE number, and the true iteration is
// index * RESIDUAL_STRIDE. If this regresses, the README "residual vs
// iteration" x-axis is silently wrong (VRAM curve looks STRIDE-times faster).
TEST(ResidualHistoryCSV, VramHistoryIsStrided) {
    int dev = 0; cudaGetDeviceCount(&dev);
    if (dev == 0) GTEST_SKIP() << "no CUDA device";
    auto g = fresh(); solver::CudaJacobiSolver s;
    s.upload(g); s.solve_vram(CAP, TOL); s.download(g);

    const std::vector<double> h = s.get_history();
    expect_valid_history(h, "JacobiGPU(vram)");
    EXPECT_LT(h.back(), TOL) << "VRAM solve did not reach tolerance";

    const int iters = s.get_vram_iterations();
    constexpr int STRIDE = solver::CudaJacobiSolver::RESIDUAL_STRIDE;

    // Sparse: far fewer samples than iterations (proves striding is in effect).
    EXPECT_LT(h.size(), static_cast<size_t>(iters))
        << "VRAM history is dense — striding regressed; row index would equal "
           "true iteration, but the rest of the pipeline assumes otherwise";
    // Roughly one sample per STRIDE iterations (+/- a couple for end conditions).
    EXPECT_NEAR(static_cast<double>(h.size()),
                static_cast<double>(iters) / STRIDE, 2.0)
        << "sample count (" << h.size() << ") not ~ iters/stride ("
        << iters << "/" << STRIDE << ") — the x-axis scaling factor is wrong";
}
#endif
