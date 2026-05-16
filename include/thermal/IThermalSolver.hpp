#pragma once
#include <vector>

struct ThermalParams {
    double alpha;           // conduction coefficient
    double u;               // convection velocity m/s
    double eps;             // radiation emissivity
    double sigma = 5.67e-8; // Stefan-Boltzmann W/m^2-K^4
};

class IThermalSolver {
public:
    virtual ~IThermalSolver() = default;
    virtual void step(double dt, double dx) = 0;
    virtual std::vector<double> getTemperatureField() const = 0;
    virtual void getResult() const = 0;
};
