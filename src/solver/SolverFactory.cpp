// src/solver/SolverFactory.cpp
#include "solver/SolverFactory.hpp"
#include "solver/JacobiCPU.hpp"
#include "solver/TDMACPU.hpp"
#ifdef PHYSI_SIM_CUDA_ENABLED
#include "solver/CudaJacobiSolver.hpp"
#endif
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace physi_sim::solver {

namespace {
std::string solver_to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}
} // namespace

std::unique_ptr<ISolver> SolverFactory::create(SolverType type, HardwareBackend backend) {
    switch (type) {
    case SolverType::JACOBI:
        if (backend == HardwareBackend::CPU) return std::make_unique<JacobiCPU>();
#ifdef PHYSI_SIM_CUDA_ENABLED     
        if (backend == HardwareBackend::CUDA) return std::make_unique<CudaJacobiSolver>();
#endif        
        break;
    case SolverType::TDMA:
        if (backend == HardwareBackend::CPU) return std::make_unique<TDMACPU>();
        throw std::invalid_argument("TDMAGPU: Phase 3b");
//        break;
    }
    throw std::invalid_argument("Unknown SolverType/Backend");
}

SolverType SolverFactory::parseSolverType(const std::string& s) {
    auto low = solver_to_lower(s);
    if (low == "jacobi") return SolverType::JACOBI;
    if (low == "tdma")   return SolverType::TDMA;
    throw std::invalid_argument("Unknown solver: '" + s + "'");
}

HardwareBackend SolverFactory::parseBackend(const std::string& s) {
    auto low = solver_to_lower(s);
    if (low == "cpu")  return HardwareBackend::CPU;
    if (low == "cuda") return HardwareBackend::CUDA;
    throw std::invalid_argument("Unknown backend: '" + s + "'");
}

} // namespace physi_sim::solver
