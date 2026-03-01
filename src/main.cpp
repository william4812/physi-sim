#include "MockBackend.hpp"
#include <iostream>
#include <memory>

int main() {
    std::cout << "--- PhysiSim LBM Solver Starting (Mock Mode) ---\n";

    // 1. Dependency Injection: Create the Mock hardware backend
    auto backend = std::make_unique<MockBackend>();
    
    // 2. Initialize the lattice (using the dimensions you set)
    // Note: Ensure your MockBackend override matches your interface name
    backend->allocate(100 * 100); 

    // 3. Inject the backend into the high-level solver
//    LBMSolver solver(std::move(backend));

    // 4. Execute a few time steps
    for (int i = 0; i < 5; ++i) {
        std::cout << "Step " << i << ": ";
//        solver.step();
        std::cout << "OK\n";
    }

    std::cout << "--- Simulation Complete ---\n";
    return 0;
}
