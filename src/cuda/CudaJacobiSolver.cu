/**
 * @file CudaJacobiSolver.cu
 * @brief GPU Jacobi solver — implements physi_sim::solver::ISolver for sm_75.
 *
 * GROUNDED IN:
 *   ISolver.hpp  — step(core::Grid2D&), residual(), name()
 *   Grid2D.hpp   — layout: index = (y * nx_) + x  [line 24 and 58]
 *                          data() returns double*  [line 35]
 *                          get_nx() = columns, get_ny() = rows
 *
 * DESIGN DECISIONS — each traceable to a test or contract:
 *
 *   m_residual = 0.0 in constructor
 *     ← ResidualIsZeroBeforeAnyStep: never returns uninitialised memory
 *
 *   copy_boundary_kernel runs BEFORE jacobi_kernel
 *     ← BoundaryValuesUnchangedAfterConvergence: d_next fully consistent
 *       before download — no garbage halo overwrites Grid2D
 *
 *   index = y * nx + x  (stride = nx, number of columns)
 *     ← Grid2D.hpp line 24: operator()(x,y) = data_[(y * nx_) + x]
 *     ← Wrong stride (y*ny+x) produces garbage — caught by
 *       FieldMatchesCPUJacobiAfterConvergence
 *
 *   __restrict__ on both kernel pointers
 *     ← d_current and d_next never alias — compiler caches neighbour
 *       loads in registers instead of reloading from VRAM after each store
 *
 *   thrust::transform + thrust::max_element for L-inf residual
 *     ← Correct parallel reduction, no race condition, ships with CUDA
 *
 *   std::swap(d_current, d_next) at end of step()
 *     ← Ping-pong: next iteration reads from what we just wrote.
 *       Swaps two pointers (16 bytes) — zero data movement in VRAM.
 */

#include "solver/CudaJacobiSolver.hpp"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>

#include <algorithm>   // std::swap, std::max
#include <cmath>       // fabs
#include <stdexcept>   // std::runtime_error

// ─────────────────────────────────────────────────────────────────────────────
// Device kernels — anonymous namespace, translation-unit private
// ─────────────────────────────────────────────────────────────────────────────
namespace 
{

/**
 * Jacobi 5-point stencil kernel.
 *
 * INDEX LAYOUT — mirrors Grid2D exactly:
 *   Grid2D::operator()(x, y) → data_[(y * nx_) + x]   (Grid2D.hpp line 24)
 *   flat index here          →         y * nx  + x     (stride = nx = columns)
 *
 * THREAD MAPPING:
 *   threadIdx.x + blockIdx.x*blockDim.x → x (column), offset +1 skips halo
 *   threadIdx.y + blockIdx.y*blockDim.y → y (row),    offset +1 skips halo
 *   Threads landing on or past the opposite halo return immediately.
 *   Boundary cells are never written — ISolver contract enforced here.
 *
 * BLOCK SIZE 16×16 = 256 threads:
 *   8 warps per block → good occupancy on sm_75 (GTX 1650, 16 SMs).
 *   Spatial locality: neighbours of adjacent threads overlap →
 *   L1 cache reuse (east neighbour of thread x = west neighbour of thread x+1).
 *
 * __restrict__:
 *   Promises d_curr and d_next never alias in memory.
 *   nvcc caches the four neighbour values in registers for the life
 *   of the kernel — no redundant VRAM round-trips after each store.
 */
__global__ void jacobi_kernel(
    const double* __restrict__ d_curr,
          double* __restrict__ d_next,
    int nx, int ny)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x + 1;  // skip left halo
    const int y = blockIdx.y * blockDim.y + threadIdx.y + 1;  // skip top  halo

    if (x >= nx - 1 || y >= ny - 1) return;                   // skip right/bottom halo

    // stride = nx — must match Grid2D: index = (y * nx_) + x
    d_next[y * nx + x] = 0.25 * (
        d_curr[(y - 1) * nx + x    ] +   // north  (y-1)
        d_curr[(y + 1) * nx + x    ] +   // south  (y+1)
        d_curr[ y      * nx + x - 1] +   // west   (x-1)
        d_curr[ y      * nx + x + 1]     // east   (x+1)
    );
}

/**
 * Boundary copy kernel: stamps d_curr halo into d_next halo.
 *
 * Problem it solves:
 *   cudaMalloc does not zero-initialise device memory.
 *   After jacobi_kernel, d_next interior is correct but boundary
 *   slots hold uninitialised garbage. Downloading d_next to Grid2D
 *   without this step would silently corrupt the caller's boundary
 *   conditions. Running this kernel first makes d_next fully consistent.
 *
 * Launch geometry:
 *   1D grid over max(nx, ny) threads.
 *   Each thread handles one column index (top/bottom rows)
 *   and one row index (left/right columns) when k is in range.
 */
__global__ void copy_boundary_kernel(
    const double* __restrict__ d_curr,
          double* __restrict__ d_next,
    int nx, int ny)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;

    // Top row (y=0) and bottom row (y=ny-1) — iterate over x
    if (k < nx) {
        d_next[0        * nx + k] = d_curr[0        * nx + k];
        d_next[(ny - 1) * nx + k] = d_curr[(ny - 1) * nx + k];
    }
    // Left col (x=0) and right col (x=nx-1) — iterate over y
    if (k < ny) {
        d_next[k * nx + 0       ] = d_curr[k * nx + 0       ];
        d_next[k * nx + (nx - 1)] = d_curr[k * nx + (nx - 1)];
    }
}

/**
 * Element-wise |a - b| functor for thrust::transform.
 * Populates d_diff_buf — the input for thrust::max_element (L-inf residual).
 * __host__ __device__: required so thrust can compile it for device execution.
 */
struct AbsDiff {
    __host__ __device__
    double operator()(double a, double b) const {
        return fabs(a - b);
    }
};

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// CudaJacobiSolver implementation
// ─────────────────────────────────────────────────────────────────────────────
namespace physi_sim::solver {

// Constructor — all members explicitly initialised.
// m_residual = 0.0 satisfies: ResidualIsZeroBeforeAnyStep.
CudaJacobiSolver::CudaJacobiSolver()
    : d_current (nullptr)
    , d_next    (nullptr)
    , d_diff_buf(nullptr)
    , m_nx      (0)
    , m_ny      (0)
    , m_residual(0.0)
{}

CudaJacobiSolver::~CudaJacobiSolver()
{
    free_device();
}

// ─────────────────────────────────────────────────────────────────────────────
void CudaJacobiSolver::free_device()
{
    if (d_current)  { cudaFree(d_current);  d_current  = nullptr; }
    if (d_next)     { cudaFree(d_next);     d_next     = nullptr; }
    if (d_diff_buf) { cudaFree(d_diff_buf); d_diff_buf = nullptr; }
    m_nx = 0;
    m_ny = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
void CudaJacobiSolver::allocate(int nx, int ny)
{
    // Already the right size — skip the expensive free/malloc cycle.
    if (nx == m_nx && ny == m_ny) return;

    free_device();

    const size_t bytes = static_cast<size_t>(nx) * ny * sizeof(double);

    cudaError_t err;

    err = cudaMalloc(&d_current, bytes);
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("CudaJacobiSolver: cudaMalloc d_current failed: ")
            + cudaGetErrorString(err));

    err = cudaMalloc(&d_next, bytes);
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("CudaJacobiSolver: cudaMalloc d_next failed: ")
            + cudaGetErrorString(err));

    err = cudaMalloc(&d_diff_buf, bytes);
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("CudaJacobiSolver: cudaMalloc d_diff_buf failed: ")
            + cudaGetErrorString(err));

    m_nx = nx;
    m_ny = ny;
}

// ─────────────────────────────────────────────────────────────────────────────
void CudaJacobiSolver::step(core::Grid2D& grid)
{
    // Grid2D.hpp: get_nx() = columns (nx_), get_ny() = rows (ny_)
    const int    nx    = grid.get_nx();
    const int    ny    = grid.get_ny();
    const size_t bytes = static_cast<size_t>(nx) * ny * sizeof(double);

    // ── 1. Ensure device buffers match current grid dimensions ────────────
    allocate(nx, ny);

    // ── 2. Upload host Grid2D → d_current (PCIe transfer) ─────────────────
    // Grid2D::data() [line 35] returns double* to the contiguous flat array.
    // Cost: ~0.6 ms for 100×100 at 16 GB/s PCIe bandwidth.
    // Phase 2 optimisation: keep d_current resident across the full solve loop.
    cudaMemcpy(d_current, grid.data(), bytes, cudaMemcpyHostToDevice);

    // ── 3. Copy boundary halo d_current → d_next ──────────────────────────
    // Runs BEFORE jacobi_kernel — makes d_next fully consistent.
    // Satisfies: BoundaryValuesUnchangedAfterConvergence.
    const int bnd_threads = 256;
    const int bnd_blocks  = (std::max(nx, ny) + bnd_threads - 1) / bnd_threads;
    copy_boundary_kernel<<<bnd_blocks, bnd_threads>>>(d_current, d_next, nx, ny);

    // ── 4. Jacobi interior update ──────────────────────────────────────────
    // 16×16 block: 256 threads, 8 warps — good sm_75 occupancy.
    // Tiles cover the full domain; threads outside interior guard with early return.
    const dim3 block(16, 16);
    const dim3 grid_dim(
        (nx + block.x - 1) / block.x,   // tiles in x (column direction)
        (ny + block.y - 1) / block.y    // tiles in y (row direction)
    );
    jacobi_kernel<<<grid_dim, block>>>(d_current, d_next, nx, ny);

    // ── 5. Synchronise — all threads must finish before residual reads ─────
    cudaDeviceSynchronize();

    // ── 6. L-inf residual: max|d_next[k] - d_current[k]| ─────────────────
    // thrust::transform builds the element-wise |diff| buffer in VRAM.
    // thrust::max_element reduces it to a single scalar on the host.
    // Correct parallel reduction — no custom kernel needed.
    thrust::device_ptr<double> p_curr(d_current);
    thrust::device_ptr<double> p_next(d_next);
    thrust::device_ptr<double> p_diff(d_diff_buf);

    thrust::transform(p_next, p_next + nx * ny,
                      p_curr,
                      p_diff,
                      AbsDiff{});

    m_residual = *thrust::max_element(p_diff, p_diff + nx * ny);
    m_history.push_back(m_residual);

    // ── 7. Download result d_next → host Grid2D ───────────────────────────
    // Same data() pointer used for upload — in-place update of Grid2D.
    cudaMemcpy(grid.data(), d_next, bytes, cudaMemcpyDeviceToHost);

    // ── 8. Ping-pong swap ──────────────────────────────────────────────────
    // d_current for next iteration = d_next from this iteration.
    // std::swap exchanges two raw pointers — 16 bytes, zero VRAM data movement.
    std::swap(d_current, d_next);
}

// ─────────────────────────────────────────────────────────────────────────────
double CudaJacobiSolver::residual() const
{
    return m_residual;
}

} // namespace physi_sim::solver
