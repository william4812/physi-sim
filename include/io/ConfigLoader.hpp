#pragma once
#include <string>
#include "core/SimulationParams.hpp"

namespace physi_sim::io 
{

/**
 * @class ConfigLoader
 * @brief Secure factory for building `SimulationParams` from untrusted input.
 *
 * @details
 * **Security Contract:** Implements an input validation boundary. The loader parses 
 * external JSON and strictly validates all fields before populating the simulation 
 * parameters.
 *
 * **Invariant:** Any missing or malformed configuration triggers an immediate 
 * runtime exception, preventing the solver from operating on uninitialized or 
 * unsafe memory bounds.
 */    
class ConfigLoader 
{
public:
    static core::SimulationParams load_json(const std::string& path);
};

} // namespace physi_sim::io 
