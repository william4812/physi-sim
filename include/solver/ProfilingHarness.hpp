// src/solver/ProfilingHarness.hpp
#pragma once
#include "solver/ISolver.hpp"
#include "core/Grid2D.hpp"
#include <memory>
#include <string>
#include <vector>

namespace physi_sim::solver {

struct ProfilingRecord {
    std::string solver_name;
    std::string backend_name;
    int         grid_nx;
    int         grid_ny;
    int         iterations;
    double      final_residual;
    double      wall_time_ms;
    bool        converged;
};

class ProfilingHarness {
public:
    explicit ProfilingHarness(std::unique_ptr<ISolver> solver);
    
    ProfilingHarness(const ProfilingHarness&)            = delete;
    ProfilingHarness& operator=(const ProfilingHarness&) = delete;
    ProfilingHarness(ProfilingHarness&&)                 = default;

    ProfilingRecord run(core::Grid2D& grid, int max_iters = 10000, double tolerance = 1e-7, bool verbose = true);
    void writeCSV(const std::string& path) const;
    const std::vector<ProfilingRecord>& results() const;
    ISolver& solver();

private:
    std::unique_ptr<ISolver>     solver_;
    std::vector<ProfilingRecord> results_;
};

} // namespace physi_sim::solver
