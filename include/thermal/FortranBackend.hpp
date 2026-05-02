#pragma once

#include "IComputeBackend.hpp"
#include <vector>

namespace physi_sim {
namespace thermal {

class FortranBackend : public IComputeBackend {
public:
    FortranBackend() = default;
    virtual ~FortranBackend() = default;

    // Standard interface for the thermal solver
    void compute(std::vector<double>& data, double alpha, double dt) override;
    void compute_steady(std::vector<double>& data, double k, double T_L, double T_R);
// --- Mandatory Interface Stubs ---
    void allocate(std::size_t size) override { /* TODO */ }
    void init(size_t w, size_t h) override { /* TODO */ }
    void collide() override { /* LBM specific - ignore for now */ }
    void stream() override { /* LBM specific - ignore for now */ }
    void applyBoundaries() override { /* TODO */ }
    void syncToHost(std::vector<double>& host_data) override { /* TODO */ }
};

} // namespace thermal
} // namespace physi_sim

