/**
 * @file CudaTDMASolver.hpp
 * @brief GPU line-by-line TDMA solver — ISolver for sm_75, VRAM-resident.
 *
 * STARTING STRATEGY (this branch): parallelize ACROSS independent lines.
 *   2D LBL-TDMA solves many independent tridiagonal systems — one per line.
 *   The Thomas algorithm is SERIAL within a line (forward + backward sweep),
 *   so the parallelism comes from the many lines: one thread per line, each
 *   running a sequential Thomas solve. Cyclic reduction / PCR (parallelism
 *   *within* a single line) is a later, optional optimization for the
 *   few-large-systems regime — NOT needed to be correct or fast here.
 *
 * Mirrors CudaJacobiSolver's lifecycle: upload -> solve_vram -> download,
 * with PCIe paid once per solve rather than once per iteration.
 *
 * THREAD SAFETY: not thread-safe (owns device pointers). One instance per
 * thread, as ConcurrentSolverRunner guarantees by construction.
 */
#pragma once

#include "solver/ISolver.hpp"
#include "core/Grid2D.hpp"
#include <string>
#include <vector>

namespace physi_sim::solver
{

class CudaTDMASolver : public ISolver
{
public:
    CudaTDMASolver();
    ~CudaTDMASolver() override;

    // Non-copyable (owns raw device memory); movable (transfers ownership).
    CudaTDMASolver(const CudaTDMASolver&)            = delete;
    CudaTDMASolver& operator=(const CudaTDMASolver&) = delete;
    CudaTDMASolver(CudaTDMASolver&&) noexcept;
    CudaTDMASolver& operator=(CudaTDMASolver&&) noexcept;

    // ── ISolver contract (match CudaJacobiSolver's signatures) ──────────
    void step(core::Grid2D& grid) override;                  ///< one LBL-TDMA sweep
    double residual() const override;                        ///< L-inf from last step()
    std::string name() const override { return "TDMAGPU"; }  ///< SolverFactory key
    const std::vector<double>& get_history() const override;

    // ── VRAM-resident lifecycle (mirror CudaJacobiSolver) ───────────────
    void upload(const core::Grid2D& grid);              ///< H2D once
    void solve_vram(int max_iter, double tolerance);    ///< full loop on device
    void download(core::Grid2D& grid) const;            ///< D2H once
    int get_vram_iterations() const noexcept { return m_vram_iterations; }

private:
    void allocate(int nx, int ny);            // size field + Thomas workspace
    void free_device();
    void check_vram_ready(const char* caller) const;
    double reduce_max_abs_diff(const double* a, const double* b, std::size_t n) const;

    // ── Device buffers ──────────────────────────────────────────────────
    // The field plus the Thomas per-line scratch. DESIGN NOTE (rung 3):
    // the forward sweep needs scratch for the modified super-diagonal (c')
    // and RHS (d') per line — choose their layout so adjacent threads touch
    // adjacent memory under your column-major storage.
    //double* d_field  = nullptr;   // device temperature field
    //double* d_cprime = nullptr;   // Thomas forward-sweep scratch (c')
    //double* d_dprime = nullptr;   // Thomas forward-sweep scratch (d')
    // ── Device buffers ──────────────────────────────────────────────────
    // Two field buffers: read neighbors from d_curr, write solved rows to
    // d_next, swap each sweep — a race-free line-Jacobi iteration.
    double* d_curr = nullptr;
    double* d_next = nullptr;

    // Per-row Thomas scratch — each row owns the slice [j*nx .. j*nx+nx).
    // The forward sweep mutates b and d, so every line needs its own copy;
    // giving a/c/x per-row too keeps the verified thomas_solve unchanged.
    double* d_a = nullptr;
    double* d_b = nullptr;
    double* d_c = nullptr;
    double* d_d = nullptr;
    double* d_x = nullptr;

    int    m_nx = 0;              // tracks allocation size to avoid realloc
    int    m_ny = 0;
    double m_residual = 0.0;      // 0.0 until first step()

    std::vector<double> m_history;   // residual per step — for CSV export

    bool m_vram_resident   = false;
    int  m_vram_iterations = 0;
};

} // namespace physi_sim::solver
