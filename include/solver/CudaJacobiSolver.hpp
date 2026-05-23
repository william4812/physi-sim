#pragma once

/**
 * @file CudaJacobiSolver.hpp
 * @brief GPU-accelerated Jacobi solver — implements ISolver for sm_75.
 *
 * Written AFTER the test file. The test file defines what this class
 * must do; this header defines how it is declared.
 *
 * CONTRACT (enforced by test_cuda_jacobi_solver.cpp):
 *   - Inherits ISolver                  (ImplementsISolverInterface)
 *   - name() returns "JacobiGPU"        (NameReturnsJacobiGPU)
 *   - residual() == 0.0 before step()   (ResidualIsZeroBeforeAnyStep)
 *   - Non-copyable                      (IsNonCopyable)
 *   - step() updates interior only      (BoundaryValuesUnchangedAfterConvergence)
 *   - Converges to same field as CPU    (FieldMatchesCPUJacobiAfterConvergence)
 *
 * MEMORY MODEL:
 *   Two ping-pong device buffers (d_current / d_next) are allocated once
 *   on the first step() call and reused for the solver's lifetime.
 *   Host Grid2D is uploaded once per step and downloaded once per step.
 *   Phase 2 optimisation: keep buffers resident across the full solve loop.
 *
 * LAYOUT CONTRACT:
 *   Grid2D uses row-major layout: index = (y * nx_) + x  (Grid2D.hpp line 24).
 *   The CUDA kernel uses the same stride. Any change to Grid2D's layout
 *   must be reflected in the kernel's index expression.
 */

#include "solver/ISolver.hpp"
#include "core/Grid2D.hpp"
#include <string>
#include <vector> // m_history

namespace physi_sim::solver 
{

class CudaJacobiSolver : public ISolver 
{
public:
    CudaJacobiSolver();
    ~CudaJacobiSolver() override;

    // Non-copyable — owns exclusive raw device memory.
    // Copying would alias CUDA pointers; double-free is UB.
    CudaJacobiSolver(const CudaJacobiSolver&)            = delete;
    CudaJacobiSolver& operator=(const CudaJacobiSolver&) = delete;

    // Movable — transfer ownership of device buffers cleanly.
    // with this:
    CudaJacobiSolver(CudaJacobiSolver&& other) noexcept
        : d_current (other.d_current)
        , d_next    (other.d_next)
        , d_diff_buf(other.d_diff_buf)
        , m_nx      (other.m_nx)
        , m_ny      (other.m_ny)
        , m_residual(other.m_residual)
        , m_history (std::move(other.m_history))
    {
        other.d_current = other.d_next = other.d_diff_buf = nullptr;
    }

    CudaJacobiSolver& operator=(CudaJacobiSolver&& other) noexcept
    {
        if (this == &other) return *this;
        free_device();
        d_current  = other.d_current;
        d_next     = other.d_next;
        d_diff_buf = other.d_diff_buf;
        m_nx       = other.m_nx;
        m_ny       = other.m_ny;
        m_residual = other.m_residual;
        m_history  = std::move(other.m_history);
        other.d_current = other.d_next = other.d_diff_buf = nullptr;
        return *this;
    }

    /**
     * One Jacobi iteration on the interior of grid.
     * Boundary cells (halo) are read but never written.
     * Uploads grid to VRAM, runs kernel, downloads result.
     */
    void step(core::Grid2D& grid) override;

    /**
     * L-inf residual from the most recent step().
     * max|T_new - T_old| over all interior cells.
     * Returns 0.0 before the first step() — never uninitialised.
     */
    double residual() const override;

    /**
     * Human-readable name for ProfilingHarness CSV and logging.
     * Must return "JacobiGPU" — test enforces the exact string.
     */
    std::string name() const override { return "JacobiGPU"; }

        /**
     * Per-step residual history — one entry per step() call.
     * history()[0] = residual after step 1, history()[n-1] = final.
     * Empty before the first step(). Use to write GPU convergence CSV.
     * Same format as jacobi_convergence.csv — directly plottable.
     */
    [[nodiscard]] const std::vector<double>& history() const { return m_history; }
private:
    // Allocate / reallocate device buffers if grid size changed.
    void allocate(int nx, int ny);

    // Free all device memory and reset dimension tracking.
    void free_device();

    double* d_current  = nullptr;   // device: T_old — read by kernel
    double* d_next     = nullptr;   // device: T_new — written by kernel
    double* d_diff_buf = nullptr;   // device: |T_new - T_old| for L-inf reduction

    int    m_nx       = 0;          // tracks allocation size to avoid realloc
    int    m_ny       = 0;
    double m_residual = 0.0;        // 0.0 until first step() — never garbage

    std::vector<double> m_history;   // residual per step — for CSV export
};

} // namespace physi_sim::solver
