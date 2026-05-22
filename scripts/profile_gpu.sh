#!/usr/bin/env bash
# =============================================================================
# profile_gpu.sh — CUDA kernel profiling using NVIDIA Nsight Compute (ncu)
#
# USAGE:
#   ./scripts/profile_gpu.sh [test_filter]
#
# EXAMPLES:
#   ./scripts/profile_gpu.sh                                # all GPU tests
#   ./scripts/profile_gpu.sh "CudaJacobiSolverTest.*"       # Jacobi GPU only
#   ./scripts/profile_gpu.sh "CudaJacobiSolverTest.Converges*"
#
# OUTPUT SECTIONS:
#   Level 1 — Occupancy and memory throughput summary
#   Level 2 — L1/L2 cache hit rates (equivalent of perf stat for GPU)
#   Level 3 — Roofline: compute vs memory bound diagnosis
#
# READING THE OUTPUT:
#   Memory Throughput % of peak:
#     > 60%  = memory-bound (expected for stencil kernels like Jacobi)
#     < 20%  = compute-bound or poorly utilised
#   L1 hit rate > 80% = good spatial locality (16×16 tiling is working)
#   Achieved Occupancy > 50% = enough warps in flight for latency hiding
#
# REQUIRES:
#   ncu (ships with CUDA toolkit >= 11.0)
#   Sudo may be required: sudo ./scripts/profile_gpu.sh
#   Or set: echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
#
# PORTABLE:
#   Copy to any CUDA project. Change BINARY and TEST_FILTER.
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
BINARY="${BUILD_DIR}/physi_tests"
TEST_FILTER="${1:-CudaJacobiSolverTest.ConvergesToTolerance}"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -f "${BINARY}" ]]; then
    echo "[ERROR] Binary not found: ${BINARY}"
    echo "        Run: cmake --build build -j\$(nproc) first"
    exit 1
fi

if ! command -v ncu &>/dev/null; then
    echo "[ERROR] ncu not found."
    echo "        ncu ships with the CUDA toolkit."
    echo "        Check: ls /usr/local/cuda/bin/ncu"
    exit 1
fi

if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] No NVIDIA GPU detected. Cannot profile."
    exit 1
fi

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Filter: ${TEST_FILTER}"
echo ""

# ── Level 1: High-level summary ───────────────────────────────────────────────
echo "======================================================="
echo "  Level 1: Occupancy + Memory Throughput Summary"
echo "  Achieved occupancy > 50% = good latency hiding"
echo "  Memory throughput > 60%  = memory-bound (normal for Jacobi)"
echo "======================================================="

ncu --set summary \
    "${BINARY}" "--gtest_filter=${TEST_FILTER}" \
    2>&1 | grep -E "(Memory|Compute|Occupancy|Duration|Kernel|Warning)" \
    | head -40

echo ""

# ── Level 2: Cache hierarchy hit rates ────────────────────────────────────────
echo "======================================================="
echo "  Level 2: L1/L2 Cache Hit Rates"
echo "  L1 hit rate > 80% = 16x16 tiling working correctly"
echo "  L1 hit rate < 50% = consider shared memory tiling (Phase 3)"
echo "======================================================="

ncu --metrics \
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    lts__t_bytes_equiv_l1sectmiss_pipe_lsu_mem_global_op_ld.sum,\
    lts__t_bytes_pipe_lsu_mem_global_op_ld.sum \
    "${BINARY}" "--gtest_filter=${TEST_FILTER}" \
    2>&1 | grep -E "(l1tex|lts|Kernel|Warning)" \
    | head -20

echo ""

# ── Level 3: Roofline diagnosis ───────────────────────────────────────────────
echo "======================================================="
echo "  Level 3: Roofline — Compute vs Memory Bound"
echo "  GTX 1650 peak compute:   ~2900 GFLOPS (FP32)"
echo "  GTX 1650 peak bandwidth: ~192  GB/s"
echo "  Jacobi arithmetic intensity: 0.125 FLOP/byte"
echo "  Expected ceiling: 192 x 0.125 = 24 GFLOPS (memory-bound)"
echo "======================================================="

ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__throughput.avg.pct_of_peak_sustained_active \
    "${BINARY}" "--gtest_filter=${TEST_FILTER}" \
    2>&1 | grep -E "(throughput|Kernel|Warning)" \
    | head -20

echo ""
echo "[INFO] GPU profiling complete."
echo "[INFO] To profile a single step (not full convergence) run:"
echo "       ./scripts/profile_gpu.sh 'CudaJacobiSolverTest.ResidualIsFiniteAfterOneStep'"
echo ""
echo "[INFO] To allow profiling without sudo:"
echo "       echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid"
