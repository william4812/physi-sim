// src/solver/TDMACPU.hpp
#pragma once
#include "solver/ISolver.hpp"

namespace physi_sim::solver 
{

class TDMACPU : public ISolver 
{
public:
    void        step(core::Grid2D& grid) override;
    double      residual() const override;
    std::string name()     const override;

    //const std::vector<double>& history() const override { return history_; }

    std::vector<double> get_history() const override;
private:
    double residual_ = 0.0;
    std::vector<double> history_; // Track it here
};

} // namespace physi_sim::solver
