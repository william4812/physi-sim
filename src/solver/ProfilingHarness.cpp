// src/solver/ProfilingHarness.cpp
#include "solver/ProfilingHarness.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace physi_sim::solver {

ProfilingHarness::ProfilingHarness(std::unique_ptr<ISolver> solver)
    : solver_(std::move(solver)) {
    if (!solver_) throw std::invalid_argument("null solver");
}

ProfilingRecord ProfilingHarness::run(core::Grid2D& grid, int max_iters, double tolerance, bool verbose) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    int    iter = 0;
    double res  = std::numeric_limits<double>::max();
    for (; iter < max_iters; ++iter) {
        solver_->step(grid);
        res = solver_->residual();
        if (verbose && iter % 100 == 0)
            std::cout << "[" << solver_->name() << "] iter=" << iter << " res=" << res << "\n";
        if (res < tolerance) { ++iter; break; }
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    ProfilingRecord rec;
    rec.solver_name    = solver_->name();
    rec.backend_name   = "cpu";
    rec.grid_nx        = grid.get_nx();
    rec.grid_ny        = grid.get_ny();
    rec.iterations     = iter;
    rec.final_residual = res;
    rec.wall_time_ms   = ms;
    rec.converged      = (res < tolerance);
    results_.push_back(rec);

    std::cout << "\n=== " << solver_->name() << " ===\n"
              << "  iters=" << iter << " res=" << res
              << " time=" << ms << "ms"
              << " converged=" << (rec.converged?"YES":"NO") << "\n\n";
    return rec;
}

void ProfilingHarness::writeCSV(const std::string& path) const {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open: " + path);
    f << "solver,backend,grid_nx,grid_ny,iterations,final_residual,wall_time_ms,converged\n";
    for (const auto& r : results_)
        f << r.solver_name << "," << r.backend_name << "," << r.grid_nx << "," << r.grid_ny << ","
          << r.iterations << "," << r.final_residual << "," << r.wall_time_ms << ","
          << (r.converged ? "true" : "false") << "\n";
}

const std::vector<ProfilingRecord>& ProfilingHarness::results() const { return results_; }
ISolver& ProfilingHarness::solver() { return *solver_; }

} // namespace physi_sim::solver
