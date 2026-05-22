#!/usr/bin/env bash
# =============================================================================
# profile_cpu.sh — Cache and cycle profiling for CPU solvers
#
# USAGE:
#   ./scripts/profile_cpu.sh [test_filter]
#
# EXAMPLES:
#   ./scripts/profile_cpu.sh                          # all CPU solver tests
#   ./scripts/profile_cpu.sh "Physics2DTest.*"        # regression tests only
#   ./scripts/profile_cpu.sh "JacobiCPUTest.*"        # Jacobi only
#
# OUTPUT:
#   Prints cache miss rate, instructions, cycles, and IPC to stdout.
#   A miss rate < 5% means the solver is cache-friendly.
#   A miss rate > 20% means the working set is spilling to RAM.
#
# REQUIRES:
#   perf (linux-tools-$(uname -r))
#   Install: sudo apt install linux-tools-common linux-tools-$(uname -r)
#
# PORTABLE:
#   Copy this file to any C++/Fortran HPC project.
#   Change BINARY and TEST_FILTER to match the new project.
# =============================================================================

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
BINARY="${BUILD_DIR}/physi_tests"
TEST_FILTER="${1:-Physics2DTest.*}"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -f "${BINARY}" ]]; then
    echo "[ERROR] Binary not found: ${BINARY}"
    echo "        Run: cmake --build build -j\$(nproc) first"
    exit 1
fi

if ! command -v perf &>/dev/null; then
    echo "[ERROR] perf not found."
    echo "        Install: sudo apt install linux-tools-common linux-tools-\$(uname -r)"
    exit 1
fi

# ── Level 1: Basic cache + cycle summary ─────────────────────────────────────
echo "======================================================="
echo "  Level 1: Cache + Cycle Summary"
echo "  Filter: ${TEST_FILTER}"
echo "======================================================="

perf stat -e \
    cache-misses,\
    cache-references,\
    instructions,\
    cycles,\
    branch-misses \
    "${BINARY}" "--gtest_filter=${TEST_FILTER}" \
    2>&1 | grep -E "(cache|instructions|cycles|branch|elapsed|seconds)"

echo ""

# ── Level 2: Cache hierarchy breakdown ───────────────────────────────────────
echo "======================================================="
echo "  Level 2: Cache Hierarchy (L1 → L2 → LLC → RAM)"
echo "  Miss rate < 5%  = cache-friendly (good)"
echo "  Miss rate > 20% = working set spilling to RAM (investigate)"
echo "======================================================="

perf stat -e \
    L1-dcache-loads,\
    L1-dcache-load-misses,\
    L2-loads,\
    L2-load-misses,\
    LLC-loads,\
    LLC-load-misses \
    "${BINARY}" "--gtest_filter=${TEST_FILTER}" \
    2>&1 | grep -E "(L1|L2|LLC|elapsed)"

echo ""

# ── Level 3: Memory bandwidth estimate ───────────────────────────────────────
echo "======================================================="
echo "  Level 3: Memory Bandwidth Estimate"
echo "  LLC-load-misses × 64 bytes / elapsed_seconds = RAM bandwidth used"
echo "======================================================="

perf stat -e \
    LLC-load-misses,\
    LLC-store-misses \
    "${BINARY}" "--gtest_filter=${TEST_FILTER}" \
    2>&1 | grep -E "(LLC|elapsed|seconds)"

echo ""
echo "[INFO] Profiling complete."
echo "[INFO] To compare solvers run:"
echo "       ./scripts/profile_cpu.sh 'JacobiCPUTest.*'"
echo "       ./scripts/profile_cpu.sh 'TDMACPUTest.*'"
