#pragma once
#include <string>

namespace physi_sim::core {

/**
 * First Principle: Plain Old Data (POD)
 * This struct is the 'Contract' between the IO layer and the Solver.
 */
struct SimulationParams 
{
    double initial_temp = 0.0;
    int max_iterations = 0;
    std::string solver_type = "unknown";
};

} // namespace physi_sim::core
