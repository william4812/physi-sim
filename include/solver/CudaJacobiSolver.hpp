/**
 * @file CudaJacobiSolver.hpp
 * @brief GPU Jacobi solver — ISolver for sm_75, Phase 2: VRAM-resident solve.
 *
 *
 *     upload(grid)              — H2D once (PCIe paid once)
 *     solve_vram(max_iter, tol) — full Jacobi loop entirely on device
 *     download(grid)            — D2H once (PCIe paid once)
 *
 *   Projected 100×100 improvement (from README §5.2):
 *     Phase 1: 4,195 iters × 0.121 ms/iter = 508 ms
 *     Phase 2: 4,195 iters × 0.001 ms/iter =  ~4 ms  (18× faster than CPU)
 *
 *   step() is preserved unchanged — ISolver contract holds, existing
 *   70 unit/integration/regression tests remain green.
 *
 * RESIDUAL CADENCE (Phase 2):
 *   Computing thrust L∞ every iteration forces cudaDeviceSynchronize per step.
 *   solve_vram() computes residual every RESIDUAL_STRIDE iterations (default 50).
 *   This amortises sync overhead while keeping convergence history meaningful.
 *
 * THREAD SAFETY:
 *   Not thread-safe (same device pointers). Use one instance per thread
 *   as ConcurrentSolverRunner already guarantees by construction.
 */

#pragma once

#include "solver/ISolver.hpp"
#include "core/Grid2D.hpp"
#include <string>
#include <vector> // m_history

namespace physi_sim::solver 
{

/**
 * @class CudaJacobiSolver
 * @brief GPU-accelerated Jacobi solver utilizing persistent VRAM buffers.
 *
 * @details
 * **Resource Contract:** Follows an explicit lifetime pattern: `upload` → `solve_vram` → 
 * `download`. This minimizes PCIe bottlenecking by amortizing host-device transfer 
 * overhead to O(1) per solver run, regardless of iteration count.
 *
 * **Memory Architecture:** Device buffers are lazily allocated and persistent. 
 * This design is "edge-ready" by construction, providing deterministic allocation 
 * behavior crucial for embedded platforms like Jetson.
 */
class CudaJacobiSolver : public ISolver 
{
public:
    CudaJacobiSolver();
    ~CudaJacobiSolver() override;

    // Number of Jacobi iterations between residual evaluations in 
    // solve_vram().
    // Trade-off: lower = more accurate history, more cudaDeviceSynchronize calls.
    // 50 gives ~84 residual samples for a typical 4,195-iteration solve.
    static constexpr int RESIDUAL_STRIDE = 50;

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

    const std::vector<double>& get_history() const override;

    // Lifetime: upload → solve_vram → download.
    // Calling solve_vram() without upload() throws std::logic_error.
    // Calling upload() again on a live solve resets device state cleanly.
 
    /**
     * Upload host grid to device VRAM. Allocates device buffers if needed.
     * H2D PCIe transfer paid exactly once per solve.
     *
     * @param grid  Source grid — must remain valid until download() returns.
     * @throws std::runtime_error on CUDA allocation or transfer failure.
     */
    void upload(const core::Grid2D& grid);
 
    /**
     * Run full Jacobi solve entirely on device.
     * No host↔device transfer during the loop.
     * Residual evaluated every RESIDUAL_STRIDE iterations for history.
     *
     * @param max_iter  Maximum iteration count.
     * @param tolerance Convergence threshold on L∞ residual (absolute).
     * @throws std::logic_error if upload() was not called first.
     */
    void solve_vram(int max_iter, double tolerance);
 
    /**
     * Download converged device field to host grid.
     * D2H PCIe transfer paid exactly once per solve.
     *
     * @param grid  Destination grid — overwritten with device result.
     * @throws std::logic_error if upload() was not called first.
     */
    void download(core::Grid2D& grid) const;
 
    /**
     * Number of iterations executed by the last solve_vram() call.
     * Returns 0 if solve_vram() has not been called.
     */
    /**
     * Number of iterations executed by the last solve_vram() call.
     * Returns 0 if solve_vram() has not been called.
     */
    int get_vram_iterations() const noexcept { return m_vram_iterations; }

private:
    // Allocate / reallocate device buffers if grid size changed.
    void allocate(int nx, int ny);

    // Free all device memory and reset dimension tracking.
    void free_device();

    // ── Device buffers ────────────────────────────────────────────────────
    double* d_current  = nullptr;   // device: T_old — read by kernel
    double* d_next     = nullptr;   // device: T_new — written by kernel
    double* d_diff_buf = nullptr;   // device: |T_new - T_old| for L-inf reduction

    int    m_nx       = 0;          // tracks allocation size to avoid realloc
    int    m_ny       = 0;
    double m_residual = 0.0;        // 0.0 until first step() — never garbage

    std::vector<double> m_history;   // residual per step — for CSV export

    // Phase 2 state
    bool m_vram_resident;    // true after upload(), false after download() or free
    int  m_vram_iterations;  // iteration count from last solve_vram()
 
    // ── Helpers ───────────────────────────────────────────────────────────
    void check_vram_ready(const char* caller) const;
};

} // namespace physi_sim::solver
