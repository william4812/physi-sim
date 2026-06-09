// src/cuda/CudaTDMASolver.cu
//
// VRAM-resident GPU line-by-line TDMA. Lifecycle mirrors CudaJacobiSolver.cu
// (upload -> solve_vram -> download). The sweep is LINE-JACOBI: one thread per
// interior row reads its j-neighbors from d_curr and writes the solved line to
// d_next, then the buffers swap. Converges to the same field as the CPU's
// line-Gauss-Seidel, just in more iterations.

#include "solver/CudaTDMASolver.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cmath>

namespace physi_sim::solver
{

// File-local helpers. The unnamed namespace gives them internal linkage, so
// they stay private to this translation unit and cannot collide with same-named
// helpers in sibling .cu files (cudaCheck, for instance, is also defined in
// CudaJacobiSolver.cu). This is the file-local `static` idiom — and the only way
// to keep a *type* like AbsDiffAt private, since `static` cannot apply to a class.
namespace
{
inline void cudaCheck(cudaError_t e, const char* what) 
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA ") + what + ": "
                                 + cudaGetErrorString(e));
}

struct AbsDiffAt 
{
    const double* a;
    const double* b;
    __host__ __device__
    double operator()(std::size_t k) const { return fabs(a[k] - b[k]); }
};
} // namespace

// the method
// L-inf change between two device fields: max_k |a[k] - b[k]|.
double CudaTDMASolver::reduce_max_abs_diff(const double* a, const double* b,
                                           std::size_t n) const
{
    return thrust::transform_reduce(
        thrust::device,                                   // run on the GPU — see note
        thrust::counting_iterator<std::size_t>(0),
        thrust::counting_iterator<std::size_t>(n),
        AbsDiffAt{a, b},
        0.0,
        thrust::maximum<double>());
}

// ───────────────── core Thomas (one tridiagonal system) ─────────────────────
// In-place variant mirroring Fortran solve_tdma: b and d are mutated.
__device__ void thomas_solve(const double* a, double* b,
                             const double* c, double* d,
                             double* x, int n)
{
    for (int i = 1; i < n; ++i) {
        double m = a[i] / b[i-1];
        b[i] -= m * c[i-1];
        d[i] -= m * d[i-1];
    }
    x[n-1] = d[n-1] / b[n-1];
    for (int i = n-2; i >= 0; --i)
        x[i] = (d[i] - c[i] * x[i+1]) / b[i];
}

// 1-thread wrapper so the unit test can exercise thomas_solve in isolation.
__global__ void thomas_single_kernel(const double* a, double* b,
                                     const double* c, double* d,
                                     double* x, int n)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        thomas_solve(a, b, c, d, x, n);
}

// Test seam (Rung 1): solve ONE system on the device.
void tdma_solve_single(const double* a, const double* b, const double* c,
                       const double* d, double* x, int n)
{
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    double *da, *db, *dc, *dd, *dx;
    cudaCheck(cudaMalloc(&da, bytes), "malloc da");
    cudaCheck(cudaMalloc(&db, bytes), "malloc db");
    cudaCheck(cudaMalloc(&dc, bytes), "malloc dc");
    cudaCheck(cudaMalloc(&dd, bytes), "malloc dd");
    cudaCheck(cudaMalloc(&dx, bytes), "malloc dx");

    cudaCheck(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice), "H2D da");
    cudaCheck(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice), "H2D db");
    cudaCheck(cudaMemcpy(dc, c, bytes, cudaMemcpyHostToDevice), "H2D dc");
    cudaCheck(cudaMemcpy(dd, d, bytes, cudaMemcpyHostToDevice), "H2D dd");

    thomas_single_kernel<<<1, 1>>>(da, db, dc, dd, dx, n);
    cudaCheck(cudaGetLastError(), "single kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "single kernel sync");

    cudaCheck(cudaMemcpy(x, dx, bytes, cudaMemcpyDeviceToHost), "D2H x");

    cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dd); cudaFree(dx);
}

// ───────────────────── Rung-2 batched line sweep ────────────────────────────
// One thread per interior row j. Implicit in i (tridiagonal solve along the
// row), explicit in j (the j-1 / j+1 neighbors are read from d_curr — that is
// what makes this a Jacobi sweep). The solved line is written into d_next.
__global__ void tdma_sweep_kernel(const double* d_curr, double* d_next,
                                  double* d_a, double* d_b, double* d_c,
                                  double* d_d, double* d_x,
                                  int nx, int ny)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < 1 || j > ny - 2) return;            // interior rows only

    double* a = d_a + j * nx;                    // this row's scratch slices
    double* b = d_b + j * nx;
    double* c = d_c + j * nx;
    double* d = d_d + j * nx;
    double* x = d_x + j * nx;

    for (int i = 0; i < nx; ++i) {
        if (i == 0 || i == nx - 1) {             // left/right walls: Dirichlet identity
            a[i] = 0.0; b[i] = 1.0; c[i] = 0.0;
            d[i] = d_curr[j * nx + i];
        } else {                                  // interior: -T(i-1)+4T(i)-T(i+1) = T(i,j-1)+T(i,j+1)
            a[i] = -1.0; b[i] = 4.0; c[i] = -1.0;
            d[i] = d_curr[(j - 1) * nx + i] + d_curr[(j + 1) * nx + i];
        }
    }

    thomas_solve(a, b, c, d, x, nx);

    for (int i = 0; i < nx; ++i)
        d_next[j * nx + i] = x[i];
}

// ───────────────────────── ctor / dtor ─────────────────────────
CudaTDMASolver::CudaTDMASolver()  = default;
CudaTDMASolver::~CudaTDMASolver() { free_device(); }

// ───────────────────────── move semantics ─────────────────────────
CudaTDMASolver::CudaTDMASolver(CudaTDMASolver&& o) noexcept
    : d_curr(o.d_curr), d_next(o.d_next)
    , d_a(o.d_a), d_b(o.d_b), d_c(o.d_c), d_d(o.d_d), d_x(o.d_x)
    , m_nx(o.m_nx), m_ny(o.m_ny), m_residual(o.m_residual)
    , m_history(std::move(o.m_history))
    , m_vram_resident(o.m_vram_resident), m_vram_iterations(o.m_vram_iterations)
{
    o.d_curr = o.d_next = o.d_a = o.d_b = o.d_c = o.d_d = o.d_x = nullptr;
}

CudaTDMASolver& CudaTDMASolver::operator=(CudaTDMASolver&& o) noexcept
{
    if (this == &o) return *this;
    free_device();
    d_curr = o.d_curr; d_next = o.d_next;
    d_a = o.d_a; d_b = o.d_b; d_c = o.d_c; d_d = o.d_d; d_x = o.d_x;
    m_nx = o.m_nx; m_ny = o.m_ny; m_residual = o.m_residual;
    m_history = std::move(o.m_history);
    m_vram_resident = o.m_vram_resident; m_vram_iterations = o.m_vram_iterations;
    o.d_curr = o.d_next = o.d_a = o.d_b = o.d_c = o.d_d = o.d_x = nullptr;
    return *this;
}

// ───────────────────────── allocation ─────────────────────────
void CudaTDMASolver::allocate(int nx, int ny)
{
    if (nx == m_nx && ny == m_ny && d_curr) return;   // already sized
    free_device();
    const size_t n = static_cast<size_t>(nx) * ny;
    cudaCheck(cudaMalloc(&d_curr, n * sizeof(double)), "malloc d_curr");
    cudaCheck(cudaMalloc(&d_next, n * sizeof(double)), "malloc d_next");
    cudaCheck(cudaMalloc(&d_a,    n * sizeof(double)), "malloc d_a");
    cudaCheck(cudaMalloc(&d_b,    n * sizeof(double)), "malloc d_b");
    cudaCheck(cudaMalloc(&d_c,    n * sizeof(double)), "malloc d_c");
    cudaCheck(cudaMalloc(&d_d,    n * sizeof(double)), "malloc d_d");
    cudaCheck(cudaMalloc(&d_x,    n * sizeof(double)), "malloc d_x");
    m_nx = nx; m_ny = ny;
}

void CudaTDMASolver::free_device()
{
    cudaFree(d_curr); d_curr = nullptr;
    cudaFree(d_next); d_next = nullptr;
    cudaFree(d_a);    d_a    = nullptr;
    cudaFree(d_b);    d_b    = nullptr;
    cudaFree(d_c);    d_c    = nullptr;
    cudaFree(d_d);    d_d    = nullptr;
    cudaFree(d_x);    d_x    = nullptr;
    m_nx = m_ny = 0;
    m_vram_resident = false;
}

void CudaTDMASolver::check_vram_ready(const char* caller) const
{
    if (!m_vram_resident)
        throw std::logic_error(std::string(caller) + ": call upload() first");
}

// ───────────────────────── lifecycle ─────────────────────────
void CudaTDMASolver::upload(const core::Grid2D& grid)
{
    const int nx = grid.get_nx();
    const int ny = grid.get_ny();
    allocate(nx, ny);
    const size_t bytes = static_cast<size_t>(nx) * ny * sizeof(double);
    cudaCheck(cudaMemcpy(d_curr, grid.data(), bytes, cudaMemcpyHostToDevice),
              "H2D d_curr");
    m_vram_resident = true;
}

void CudaTDMASolver::solve_vram(int max_iter, double tolerance)
{
    check_vram_ready("solve_vram");
    m_history.clear();
    m_vram_iterations = 0;

    const size_t bytes = static_cast<size_t>(m_nx) * m_ny * sizeof(double);
    const int threads = 256;
    const int blocks  = (m_ny + threads - 1) / threads;

    for (int it = 0; it < max_iter; ++it)
    {
        // Carry the unchanged boundaries + top/bottom rows into d_next; the
        // kernel overwrites only the interior rows.
        cudaCheck(cudaMemcpy(d_next, d_curr, bytes, cudaMemcpyDeviceToDevice),
                  "D2D carry");
        tdma_sweep_kernel<<<blocks, threads>>>(d_curr, d_next,
                                               d_a, d_b, d_c, d_d, d_x,
                                               m_nx, m_ny);
        cudaCheck(cudaGetLastError(), "sweep launch");

        double* tmp = d_curr; d_curr = d_next; d_next = tmp;   // result now in d_curr
        ++m_vram_iterations;

        // RUNG 4: every RESIDUAL_STRIDE iters, reduce max|d_curr - d_next|,
        // push to m_history, and break when below tolerance.
        (void)tolerance;
    }
    cudaCheck(cudaDeviceSynchronize(), "solve_vram sync");
}

void CudaTDMASolver::download(core::Grid2D& grid) const
{
    check_vram_ready("download");
    const size_t bytes = static_cast<size_t>(m_nx) * m_ny * sizeof(double);
    cudaCheck(cudaMemcpy(grid.data(), d_curr, bytes, cudaMemcpyDeviceToHost),
              "D2H d_curr");
}

// ───────────────────────── ISolver per-step path ─────────────────────────
// One sweep with PCIe each call — exists only to satisfy ISolver. The fast
// path is upload / solve_vram / download.
void CudaTDMASolver::step(core::Grid2D& grid)
{
    upload(grid);
    const size_t bytes = static_cast<size_t>(m_nx) * m_ny * sizeof(double);
    const int threads = 256;
    const int blocks  = (m_ny + threads - 1) / threads;

    cudaCheck(cudaMemcpy(d_next, d_curr, bytes, cudaMemcpyDeviceToDevice), "D2D step");
    tdma_sweep_kernel<<<blocks, threads>>>(d_curr, d_next,
                                           d_a, d_b, d_c, d_d, d_x,
                                           m_nx, m_ny);
    cudaCheck(cudaGetLastError(), "step kernel");
    double* tmp = d_curr; d_curr = d_next; d_next = tmp;
    cudaCheck(cudaDeviceSynchronize(), "step sync");
    
    // RUNG 4: set m_residual + push to m_history for this step.
    const std::size_t n = static_cast<std::size_t>(m_nx) * m_ny;
    m_residual = reduce_max_abs_diff(d_curr, d_next, n);   // d_curr=NEW, d_next=OLD after the swap
    
    download(grid);
}

double CudaTDMASolver::residual() const { return m_residual; }

const std::vector<double>& CudaTDMASolver::get_history() const { return m_history; }

} // namespace physi_sim::solver
