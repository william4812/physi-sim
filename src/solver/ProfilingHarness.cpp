// src/solver/ProfilingHarness.cpp
#include "solver/ProfilingHarness.hpp"
#include "core/LaplaceResidual.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace physi_sim::solver 
{

ProfilingHarness::ProfilingHarness(std::unique_ptr<ISolver> solver) : solver_(std::move(solver)) 
{
    if (!solver_) 
        throw std::invalid_argument("null solver");
    // fsm_ initialises to SolverState::IDLE automatically.
}

// ── run() ─────────────────────────────────────────────────────────────────────
// FSM lifecycle:
//   IDLE → prepare() → READY → start() → RUNNING
//       → finish(true|false) → CONVERGED | FAILED
//       → [capture fsm_state] → reset() → IDLE
ProfilingRecord ProfilingHarness::run(core::Grid2D& grid, 
                                      int max_iters, 
                                      double tolerance, 
                                      bool verbose) 
{
    // FSM: IDLE → READY → RUNNING
    // prepare() throws std::logic_error if not IDLE — guards re-entrancy.
    fsm_.prepare();   // IDLE → READY
    fsm_.start();     // READY → RUNNING
 
    // Timer starts after FSM bookkeeping — we time physics, not state ops.
    const auto t0 = std::chrono::high_resolution_clock::now();
 
    int    iter             = 0;
    double res              = std::numeric_limits<double>::max();
    double residual_initial = -1.0;   // set on iter 0, used for normalization 
    
    for (; iter < max_iters; ++iter) 
    {

        solver_->step(grid); // INCREMENT ||T^k - T^{k-1}||_inf
        res = solver_->residual();
        if (iter == 0) residual_initial = res;   // / res_0 := first increment (anchor)
       
        // Relative progress of the INCREMENT: res/res_0, dimensionless — the LIVE
        // stopping test. For Jacobi the 1/4 cancels in the ratio, so the
        // normalized increment equals the normalized equation residual; that is
        // why same-algorithm (CPU vs GPU) comparison is already fair on the
        // increment. CROSS-algorithm fairness needs the equation residual below. 
        if (verbose && iter % 100 == 0)
            std::cout << "[" << solver_->name() << "]"
                      << " iter=" << iter
                      << " res="  << std::scientific << std::setprecision(3)
                      << res << "\n";

        const double normalized = (residual_initial > 0.0)
            ? res / residual_initial : res;

        if (normalized < tolerance)
        {
            fsm_.finish(true);   // RUNNING → CONVERGED
            ++iter;
            break;
        }
    }
 
    // Max iters exhausted without convergence
    if (fsm_.state() == SolverState::RUNNING)
        fsm_.finish(false);      // RUNNING → FAILED
 
    const double wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
 
    // Capture terminal state BEFORE reset() writes IDLE.
    const bool        converged = (fsm_.state() == SolverState::CONVERGED);
    const std::string fsm_state = std::string(fsm_.stateName());
 
    // Reset: harness ready for next run() call.
    fsm_.reset();     // CONVERGED | FAILED → IDLE
 
    // Assemble record
    ProfilingRecord rec;
    rec.solver_name    = solver_->name();
    rec.backend_name = (solver_->name().find("GPU") 
            != std::string::npos) ? "cuda" : "cpu";
    rec.grid_nx        = grid.get_nx();
    rec.grid_ny        = grid.get_ny();
    rec.iterations     = iter;
    rec.final_residual = res;
    rec.normalized_residual = (residual_initial > 0.0)
                                ? res / residual_initial
                                : res;
     // UNIFIED, algorithm-independent residual ||b - A.T||_inf of the CONVERGED
    // field — the apples-to-apples number for Jacobi vs TDMA vs GPU. Computed
    // ONCE, AFTER the timer stopped (one stencil pass, zero timing impact), on
    // the field every solver leaves in `grid`. Operator correctness is locked by
    // the ResidualOperator.* tests; the 4x tie to the increment by
    // Convergence.EquationEqualsFourTimesIncrementEveryStep.
    rec.equation_residual = core::laplace_residual_linf(grid);   // NEW: measure converged field
    rec.wall_time_ms   = wall_ms;
    rec.converged      = converged;
    rec.fsm_state      = fsm_state;
 
    results_.push_back(rec);
 
    std::cout << "\n=== " << rec.solver_name << " ===\n"
              << "  iters=" << rec.iterations
              << " res="    << std::scientific << std::setprecision(3)
                            << rec.final_residual
              << " time="   << std::fixed << std::setprecision(3)
                            << rec.wall_time_ms << "ms"
              << " fsm="    << rec.fsm_state << "\n\n";
 
    return rec;
}

// ── writeCSV() ────────────────────────────────────────────────────────────────
// fsm_state is the last column — existing consumers that stop reading at
// "converged" are not broken by this addition.
//
void ProfilingHarness::writeCSV(const std::string& path) const 
{
    std::ofstream f(path);

    if (!f) throw std::runtime_error("cannot open: " + path);

       // Header — fsm_state added at end
    //f << "solver,backend,grid_nx,grid_ny,"
    //  << "iterations,final_residual,normalized_residual,wall_time_ms,converged,fsm_state\n"; 
    f << "solver,backend,grid_nx,grid_ny,"
      << "iterations,final_residual,normalized_residual,equation_residual,"
      << "wall_time_ms,converged,fsm_state\n";

    for (const auto& r : results_)
    {
        f << r.solver_name    << ","
          << r.backend_name   << ","
          << r.grid_nx        << ","
          << r.grid_ny        << ","
          << r.iterations     << ","
          << std::scientific << std::setprecision(6) << r.final_residual << ","
          << std::scientific << std::setprecision(6) << r.normalized_residual << ","
          << std::scientific << std::setprecision(6) << r.equation_residual   << ","
          << std::fixed      << std::setprecision(4) << r.wall_time_ms   << ","
          << (r.converged ? "true" : "false")         << ","
          << r.fsm_state      << "\n";
    }
}

// ── Accessors ─────────────────────────────────────────────────────────────────

const std::vector<ProfilingRecord>& ProfilingHarness::results() const 
{
    return results_;
}

ISolver& ProfilingHarness::solver() 
{ 
    return *solver_; 
}

SolverState ProfilingHarness::fsmState() const noexcept 
{
    return fsm_.state();
}

} // namespace physi_sim::solver
