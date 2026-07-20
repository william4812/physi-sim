#include "thermal3d/ElectroThermal3D.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace physi_sim::thermal3d;

// Simple test framework macro to keep dependency zero while asserting physics
TEST(ElectroThermalPhysicsTest, TSVJouleIntegralMatchesIsquaredR) 
{
    std::cout << "[RUN] TSVJouleIntegral.MatchesIsquaredR...\n";

    const std::size_t n = 16;
    const double L = 1e-3; // 1 mm cube unit cell
    const double sigma_cu = 5.8e7; // Copper conductivity [S/m]

    // Uniform copper field
    ElectroThermal3D::ScalarField sigma = [sigma_cu](double, double, double) {
        return sigma_cu;
    };

    ElectroThermal3D solver(n, L, sigma);

    // Apply 1.0V across Z-axis (ZLow = 1.0V, ZHigh = 0.0V), lateral faces insulated (Neumann 0)
    ElectroThermal3D::BCs bcs;
    for (std::size_t f = 0; f < ElectroThermal3D::FaceCount; ++f) {
        bcs[f].type = ElectroThermal3D::FaceBC::Neumann;
        bcs[f].value = [](double, double, double) { return 0.0; };
    }
    
    bcs[ElectroThermal3D::ZLow].type = ElectroThermal3D::FaceBC::Dirichlet;
    bcs[ElectroThermal3D::ZLow].value = [](double, double, double) { return 1.0; };

    bcs[ElectroThermal3D::ZHigh].type = ElectroThermal3D::FaceBC::Dirichlet;
    bcs[ElectroThermal3D::ZHigh].value = [](double, double, double) { return 0.0; };

    solver.setElectricalBC(bcs);

    // Solve to tight tolerance to verify machine-precision conservation
    bool converged = solver.solve(1e-14, 500000);
    if (!converged) {
        std::cerr << "[FAIL] Solver failed to converge within iteration limit!\n";
        std::exit(1);
    }

    // ANALYTICAL CIRCUIT PIN:
    // R = L / (sigma * A) = 1e-3 / (5.8e7 * 1e-6) = 1 / 58 Ohms (~0.017241379 Ohm)
    // I = DeltaV / R = 1.0 / (1 / 58) = 58.0 Amps
    // P = I^2 * R = (58.0)^2 * (1 / 58) = 58.0 Watts
    const double area = L * L;
    const double analytical_R = L / (sigma_cu * area);
    const double analytical_I = 1.0 / analytical_R;
    const double expected_power = analytical_I * analytical_I * analytical_R; // Exact 58.0 W

    const double actual_power = solver.totalJoulePower();

    // Assert conservation matches machine precision (~1e-13 relative error)
    const double rel_tol = 1e-11 * expected_power;
    EXPECT_NEAR(actual_power, expected_power, rel_tol) 
        << "Volume integral of sigma|grad V|^2 must match lumped I^2*R circuit dissipation exactly.";
    std::cout << "[PASS] TSVJouleIntegral.MatchesIsquaredR (Power = " << actual_power << " W)\n";
}

