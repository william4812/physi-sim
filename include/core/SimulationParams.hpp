#pragma once
#include <string>
#include <vector>

namespace physi_sim::core {

/**
 * @brief POD config struct — contract between IO layer and solver.
 *
 * Loaded by ConfigLoader from config.yaml.
 * Passed to SolverFactory and SimStateMachine.
 * All fields have safe defaults so partial configs work.
 */
struct SimulationParams {
    std::string solver_type    = "tdma";  // "jacobi" or "tdma"
    std::string backend        = "cpu";   // "cpu" or "cuda"
    int         grid_nx        = 64;      // interior points in x
    int         grid_ny        = 64;      // interior points in y
    double      tolerance      = 1e-7;    // L-inf convergence threshold
    int         max_iterations = 10000;   // iteration cap
    double      initial_temp   = 0.0;     // initial field value

    // benchmark sweep fields
    std::vector<int> grid_sizes = {100, 500};  // NX=NY for each run
    bool             run_gpu    = true;         // include JacobiGPU in sweep
    std::string      output_dir = ".";          // CSV output directory
};

} // namespace physi_sim::core
