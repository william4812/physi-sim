#pragma once

#include "IComputeBackend.hpp"
#include <vector>

// Parameters for mixed-mode heat transfer
struct ThermalParams 
{
    double alpha;  // Conduction
    double u;      //m/s Convection velocity
    double eps;    // Radiation emissivity
    double sigma = 5.67e-8;  //W/m^2-K^4, Stefan-Boltzmann
};

class IThermalSolver 
{
public:
    virtual ~IThermalSolver() = default;
    virtual void step(double dt, double dx) = 0;
    virtual std::vector<double> getTemperatureField() const = 0;
    virtual void getResult() const = 0;
};

