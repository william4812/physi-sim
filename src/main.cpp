#include "MockBackend.hpp"
#include "thermal/LinearDummySolver.hpp"
#include "io/VTKWriter.hpp"
#include <iostream>
#include <memory>

int main() {
    std::cout << "--- PhysiSim LBM Solver Starting (Mock Mode) ---\n";

    // 1. Dependency Injection: Create the Mock hardware backend
    auto backend = std::make_unique<MockBackend>();

    // 2. Setup Parameters
    const size_t domain_size = 100;
    double dt{1.0};
    double dx{1.0};

    // 3. Inject the backend into the high-level solver
    // 3. Initialize Solver
    // NOTE: You can only move 'backend' ONCE. 
    // We give it to dSolver since that is what we are testing today.
    LinearDummySolver dSolver(std::move(backend), domain_size);

    // 4. Execute time steps to fill the linear profile
    for (int i = 0; i < domain_size; ++i) {
        std::cout << "Step " << i << ": ";
        dSolver.step(dt, dx);
        std::cout << "OK\n";
    }

    // 5. Output Results
    std::cout << "Writing results to VTK...\n";
    auto field = dSolver.getTemperatureField();

    physi_sim::io::VTKWriter writer;
    writer.write(field, "thermal_results.vtk");

    std::cout << "--- Simulation Complete ---\n";
    return 0;
}
