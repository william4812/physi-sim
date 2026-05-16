// src/solver/TDMACPU.hpp
#pragma once
#include "solver/ISolver.hpp"

namespace physi_sim::solver {

class TDMACPU : public ISolver {
public:
    void        step(core::Grid2D& grid) override;
    double      residual() const override;
    std::string name()     const override;
private:
    double residual_ = 0.0;
};

} // namespace physi_sim::solver
