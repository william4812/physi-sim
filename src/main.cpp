#include "MockBackend.hpp"
#include "thermal/LinearDummySolver.hpp"
#include "thermal/FortranBackend.hpp"
#include "io/VTKWriter.hpp"
#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>

void run_thermal_benchmark(); 

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


    try 
    {
        run_thermal_benchmark();
    } catch (const std::exception& e) 
    {
        std::cerr << "Hardware/Logic Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}


/**
 * @brief Executes a high-performance thermal simulation benchmark.
 * * 1st Principle: Isolate side-effects (IO) from the compute-intensive loop
 * to ensure timing accuracy.
 */
void run_thermal_benchmark() 
{
    using namespace physi_sim::thermal;

    // Configuration: 10^6 nodes is large enough to exceed L2 cache,
    // testing the memory bandwidth of your X1 Extreme.
    const int nx = 1000000;
    const double alpha = 0.01;
    const double dt = 0.0001;
    const int iterations = 100;

    // Memory Allocation (Contiguous for SIMD optimization)
    std::vector<double> grid(nx, 0.0);
    grid[nx / 2] = 1.0; // Initial Heat Spike

    FortranBackend solver;

    std::cout << "--- Starting High-Performance Thermal Benchmark ---" << std::endl;
    std::cout << "Grid Size: " << nx << " nodes" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        solver.compute(grid, alpha, dt);
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Calculate Metrics
    std::chrono::duration<double> elapsed = end - start;
    double total_ops = 5.0 * (nx - 2) * iterations; // 5 FLOPs per interior node
    double gflops = (total_ops / elapsed.count()) / 1e9;

    // Report Results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Simulation Complete." << std::endl;
    std::cout << "Total Time:     " << elapsed.count() << " s" << std::endl;
    std::cout << "Avg per Step:   " << (elapsed.count() / iterations) * 1000.0 << " ms" << std::endl;
    std::cout << "Performance:    " << gflops << " GFLOPS" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
}
