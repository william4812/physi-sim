// src/solver/ProfilingHarness.hpp
#pragma once
#include "solver/ISolver.hpp"
#include "core/Grid2D.hpp"
#include "solver/SolverFSM.hpp"
#include <memory>
#include <string>
#include <vector>

namespace physi_sim::solver {

struct ProfilingRecord {
    std::string solver_name;
    std::string backend_name;
    int         grid_nx        = 0;
    int         grid_ny        = 0;
    int         iterations     = 0;
    double      final_residual = 0.0;
    double      normalized_residual = 0.0; 
    double      equation_residual   = 0.0;   // NEW: unified ||b - A·T||_inf on final field
    double      wall_time_ms   = 0.0;
    bool        converged      = false;
    std::string fsm_state;      // "CONVERGED" | "FAILED"  ← Phase 2 addition
};

class ProfilingHarness 
{
public:
  explicit ProfilingHarness(std::unique_ptr<ISolver> solver);

  ProfilingHarness(const ProfilingHarness&)            = delete;
  ProfilingHarness& operator=(const ProfilingHarness&) = delete;
  ProfilingHarness(ProfilingHarness&&) noexcept            = default;
  ProfilingHarness& operator=(ProfilingHarness&&) noexcept = default;

  // run() drives IDLE→READY→RUNNING→CONVERGED|FAILED→IDLE.
  // rec.fsm_state is set before reset() — never empty after this call.
  [[nodiscard]]
  ProfilingRecord run(core::Grid2D& grid,
                      int           max_iters,
                      double        tolerance,
                      bool          verbose = false);

  void writeCSV(const std::string& path) const;
  [[nodiscard]] const std::vector<ProfilingRecord>& results() const;
  [[nodiscard]] ISolver& solver();

  // Thread-safe: ConcurrentSolverRunner reads this from a monitor thread.
  [[nodiscard]] SolverState fsmState() const noexcept;

private:
  std::unique_ptr<ISolver>     solver_;
  SolverFSM                    fsm_;
  std::vector<ProfilingRecord> results_;
};

} // namespace physi_sim::solver
