// src/solver/JacobiCPU.hpp
#pragma once
#include "solver/ISolver.hpp"
#include <vector>

namespace physi_sim::solver 
{

class JacobiCPU : public ISolver 
{
public:
    void        step(core::Grid2D& grid) override;
    double      residual() const override;
    std::string name()     const override;
    // Per-step residual history — same contract as CudaJacobiSolver.
    // history()[i] = residual after step i+1.
    // Empty before first step(). Used to write convergence CSV.
    [[nodiscard]] const std::vector<double>& history() const { return history_; }
private:
    double residual_ = 0.0;
    std::vector<double> history_;
};

} // namespace physi_sim::solver
