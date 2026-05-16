#pragma once
#include <vector>

namespace physi_sim::thermal {

class FortranBackend {
public:
    FortranBackend()          = default;
    virtual ~FortranBackend() = default;
    void compute(std::vector<double>& data, double alpha, double dt);
    void compute_steady(std::vector<double>& data,
                        double k, double T_L, double T_R);
};

} // namespace physi_sim::thermal
