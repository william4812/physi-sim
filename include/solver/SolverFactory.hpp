// src/solver/SolverFactory.hpp
#pragma once
#include "solver/ISolver.hpp"
#include <memory>
#include <string>

namespace physi_sim::solver {

class SolverFactory 
{
public:
    static std::unique_ptr<ISolver> create(SolverType type, HardwareBackend backend);
    static SolverType parseSolverType(const std::string& s);
    static HardwareBackend parseBackend(const std::string& s);
};

} // namespace physi_sim::solver
