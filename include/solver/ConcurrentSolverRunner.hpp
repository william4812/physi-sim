// include/solver/ConcurrentSolverRunner.hpp
//
// Phase 3 — concurrent dispatch of independent ISolver instances.
//
// DESIGN: static run() — no instance state, no lifecycle to manage.
// Matches the calling convention of a Fortran `pure` subroutine or a
// CUDA kernel launch: inputs in, outputs out, zero side effects.
//
// MEMORY ISOLATION:
//   const Grid2D& grid_template communicates intent: this is read-only.
//   The runner copies it once per solver thread. Each thread owns a
//   flat vector<double> — independent heap allocation, no aliasing.
//
//   CPU effect:  stride-1 Fortran inner loop (do i=2,nx-1) runs at
//                full L1/L2 bandwidth on each thread's own cache lines.
//   GPU preview: same flat layout maps to coalesced global memory access
//                when CudaThermalSolver (Phase 5) joins the dispatch.
//
// THREAD SAFETY:
//   SolverFSM uses std::atomic<SolverState> — fsmState() readable from
//   any thread without a lock. ProfilingHarness::run() drives one FSM
//   per thread. No shared mutable state anywhere in the dispatch path.
 
#pragma once
#include "solver/ISolver.hpp"
#include "solver/ProfilingHarness.hpp"
#include "core/Grid2D.hpp"
#include <vector>
#include <memory>

namespace physi_sim::solver 
{

struct RunConfig
{
    int    max_iters = 1000;
    double tolerance = 1e-6;
    bool   verbose   = false;
};

class ConcurrentSolverRunner 
{
public:
    // Dispatch every solver in `solvers` simultaneously.
    //
    //   solvers       — runner takes ownership (unique_ptr by value)
    //   grid_template — copied once per solver; original never modified
    //   cfg           — shared read-only config for all threads
    //
    // Returns one ProfilingRecord per solver, in dispatch order.
    // Each record carries: wall_time_ms, fsm_state, solver_name, grid dims.
    //
    // Concurrent wall time < sum(record.wall_time_ms) proves overlap.
    [[nodiscard]]
    static std::vector<ProfilingRecord> run(
        std::vector<std::unique_ptr<ISolver>> solvers,
        const core::Grid2D&                   grid_template,
        const RunConfig&                      cfg = {});

    // Not instantiable — pure static utility, like SolverFactory.
    ConcurrentSolverRunner() = delete;
};

}// namespace physi_sim::solver 
