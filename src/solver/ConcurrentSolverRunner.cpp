// src/solver/ConcurrentSolverRunner.cpp
//
// IMPLEMENTATION PRINCIPLE:
//   Race-free by memory isolation, not by lock.
//   Each lambda captures:
//     h — ProfilingHarness (owns solver + SolverFSM)  → thread-local
//     g — Grid2D copy                                  → thread-local
//   cfg captured by reference — const, lives on caller's stack
//   throughout run(). Safe to read from N threads simultaneously.
//
// WHY THE FLAT ARRAY MATTERS:
//   Grid2D::data_ = std::vector<double> = contiguous heap allocation.
//   `Grid2D g = grid_template` copies nx_, ny_, and the entire vector.
//   Independent allocations — no cache-line sharing between threads.
//
//   Fortran inner loop: do i = 2, nx-1
//     i is the fast (column) index in Fortran column-major order.
//     Each thread's copy has its own base address.
//     Prefetcher loads 8 consecutive doubles per 64-byte cache line.
//     At full bandwidth: memory-bandwidth-bound, not compute-bound.
//
//   CUDA phase 5 preview:
//     flat[i + j*nx] with i = threadIdx.x
//     Consecutive threads access consecutive addresses = coalesced.
//     Same stride-1 principle, different hardware unit (HBM not LLC).
 
#include "solver/ConcurrentSolverRunner.hpp"
#include <future>
#include <vector>

namespace physi_sim::solver 
{

std::vector<ProfilingRecord> ConcurrentSolverRunner::run(
        std::vector<std::unique_ptr<ISolver>> solvers,
        const core::Grid2D&                   grid_template,
        const RunConfig&                      cfg)
{
    // ── Dispatch ──────────────────────────────────────────────────────────────
    // One future per solver. Each lambda owns:
    //   h — harness (moves solver in, drives FSM lifecycle)
    //   g — independent Grid2D copy (flat vector<double>, no aliasing)
    // cfg captured by reference — const, shared safely across threads.

    std::vector<std::future<ProfilingRecord>> futures;
    futures.reserve(solvers.size());

    for (auto& solver : solvers) 
    {
        futures.emplace_back(
            std::async(
                std::launch::async,
                // Init-capture: unique_ptr moved INTO lambda, Grid2D copied INTO lambda.
                // No extra args to std::async — avoids tuple-packing unique_ptr.
                [s = std::move(solver), grid = grid_template, &cfg]() mutable 
                {
                    ProfilingHarness harness(std::move(s));
                    return harness.run(grid,
                                       cfg.max_iters,
                                       cfg.tolerance,
                                       cfg.verbose);
                }
            )
        );
    }

    // ── Collect ───────────────────────────────────────────────────────────────
    // future::get() blocks until that thread finishes and propagates exceptions.
    // Results preserved in dispatch order — deterministic output.

    std::vector<ProfilingRecord> results;
    results.reserve(futures.size());
    for (auto& f : futures) 
    {
        results.push_back(f.get());
    }

    return results;
}

}//namespace physi_sim::solver
