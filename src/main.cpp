#include "MockBackend.hpp"
#include "thermal/LinearDummySolver.hpp"
#include "thermal/FortranBackend.hpp"
#include "io/VTKWriter.hpp"
#include "io/ConfigLoader.hpp"
#include "solver/SolverFactory.hpp"      // 
#include "solver/ProfilingHarness.hpp"    
#include "core/Grid2D.hpp"               
#ifdef PHYSI_SIM_CUDA_ENABLED
#include "solver/CudaJacobiSolver.hpp"
#endif
#include "solver/JacobiCPU.hpp"
#include <fstream>                       // CSV writing
#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>

// forward declaration
void run_thermal_benchmark(); 
//auto params = physi_sim::io::ConfigLoader::load_json("../benchmark_config.json");
void run_solver_benchmark(const physi_sim::core::SimulationParams& params);             

// ── Benchmark FSM states ──────────────────────────────────────────────────────
// Mirrors SolverFSM pattern: explicit states, no implicit control flow.
enum class BenchmarkState { INIT, RUNNING, WRITING, DONE };

int main() 
{
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
    for (size_t i = 0; i < domain_size; ++i) 
    {
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
        // Load config inside main() — not at global scope.
        // Global constructors run before exception handlers exist.
        const auto params =
            physi_sim::io::ConfigLoader::load_json("../benchmark_config.json");

        run_thermal_benchmark();
        run_solver_benchmark(params);
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

/**
 * @brief Four-way ISolver benchmark: JacobiCPU, TDMACPU, JacobiGPU.
 *
 * ProfilingHarness owns the FSM lifecycle for each solver:
 *   IDLE → prepare() → READY → start() → RUNNING
 *       → finish(converged) → CONVERGED|FAILED → reset() → IDLE
 *
 * All solvers run on the same 100×100 grid with the same boundary
 * condition (T=100 top row) and the same normalized convergence
 * criterion (L∞ < 1e-4 relative to iteration-0 residual).
 *
 * Outputs:
 *   profiling_results.csv           — wall time + iterations per solver
 *   JacobiCPU_convergence.csv       — per-step residual history
 *   TDMACPU_convergence.csv         — per-step residual history
 *   JacobiGPU_convergence.csv       — per-step residual history (GPU)
 */
void run_solver_benchmark(const physi_sim::core::SimulationParams& params)
{
    using namespace physi_sim;
    using solver::SolverFactory;
    using solver::SolverType;
    using solver::HardwareBackend;
    using solver::ProfilingHarness;

    // ── Solver entries — CUDA guarded for CI builds ───────────────────────
    struct Entry { SolverType type; HardwareBackend backend; };
    const std::vector<Entry> entries = {
        { SolverType::JACOBI, HardwareBackend::CPU  },
        { SolverType::TDMA,   HardwareBackend::CPU  },
#ifdef PHYSI_SIM_CUDA_ENABLED
        { SolverType::JACOBI, HardwareBackend::CUDA },
#endif
    };

    // ── FSM ───────────────────────────────────────────────────────────────
    auto state = BenchmarkState::INIT;

    // results indexed by grid size: results_by_grid[N] = vector of records
    std::vector<std::pair<int,
        std::vector<std::unique_ptr<ProfilingHarness>>>> results_by_grid;

    while (state != BenchmarkState::DONE)
    {
        switch (state)
        {
        // ── INIT: validate and print header ──────────────────────────────
        case BenchmarkState::INIT:
            std::cout << "\n=== ISolver Grid-Sweep Benchmark ===\n";
            std::cout << "Grid sizes: ";
            for (int n : params.grid_sizes) std::cout << n << " ";
            std::cout << "\nTolerance:  " << params.tolerance
                      << "  (normalized L-inf)\n";
            std::cout << "Max iters:  " << params.max_iterations << "\n\n";
            state = BenchmarkState::RUNNING;
            break;

        // ── RUNNING: loop over all grid sizes and solvers ─────────────────
        case BenchmarkState::RUNNING:
            for (int N : params.grid_sizes)
            {
                std::cout << "--- Grid " << N << "x" << N << " ---\n";

                core::Grid2D grid_template(N, N);
                for (int x = 0; x < N; ++x)
                    grid_template(x, N - 1) = params.initial_temp;

                std::vector<std::unique_ptr<ProfilingHarness>> harnesses;

                for (const auto& e : entries)
                {
                    core::Grid2D grid = grid_template;
                    auto s = SolverFactory::create(e.type, e.backend);
                    const std::string sname = s->name();

                    auto h = std::make_unique<ProfilingHarness>(std::move(s));
                    auto rec = h->run(grid, params.max_iterations,
                                      params.tolerance, /*verbose=*/false);
                    (void)rec;

                    std::cout << "  " << sname
                              << "  iters=" << h->results().back().iterations
                              << "  time="  << h->results().back().wall_time_ms
                              << "ms\n";

                    // ── Write per-step convergence CSV ────────────────────
                    // CPU: cast to JacobiCPU to access history()
                    // GPU: cast to CudaJacobiSolver to access history()
                    // Same file naming: {solver}_{N}x{N}_convergence.csv
                    // Same format: Iteration,Residual
                    // → both CSVs are directly comparable in plots

                    const std::vector<double>* hist_ptr = nullptr;

                    // JACOBI + CPU
                    if (e.backend == HardwareBackend::CPU
                        && e.type == SolverType::JACOBI)
                    {
                        auto* cpu = dynamic_cast<
                            solver::JacobiCPU*>(&h->solver());
                        if (cpu) hist_ptr = &cpu->history();
                    }

#ifdef PHYSI_SIM_CUDA_ENABLED
                    if (e.backend == HardwareBackend::CUDA)
                    {
                        auto* gpu = dynamic_cast<
                            solver::CudaJacobiSolver*>(&h->solver());
                        if (gpu) hist_ptr = &gpu->history();
                    }
#endif

                    if (hist_ptr && !hist_ptr->empty())
                    {
                        const std::string fname =
                            sname + "_" + std::to_string(N)
                            + "x" + std::to_string(N)
                            + "_convergence.csv";
                        std::ofstream f(fname);
                        f << "Iteration,Residual\n";
                        for (int i = 0; i < (int)hist_ptr->size(); ++i)
                            f << i << ","
                              << std::scientific << (*hist_ptr)[i] << "\n";
                        std::cout << "  [IO] " << fname << "\n";
                    }

                    harnesses.push_back(std::move(h));
                }
                results_by_grid.emplace_back(N, std::move(harnesses));
            }
            state = BenchmarkState::WRITING;
            break;

        // ── WRITING: flush all CSVs ───────────────────────────────────────
        case BenchmarkState::WRITING:
            for (auto& [N, harnesses] : results_by_grid)
            {
                const std::string fname =
                    std::to_string(N) + "x" + std::to_string(N)
                    + "_profiling_results.csv";
                std::ofstream f(fname);
                f << "solver,backend,grid_nx,grid_ny,"
                  << "iterations,final_residual,normalized_residual,"
                  << "wall_time_ms,converged,fsm_state\n";
                for (auto& h : harnesses)
                    for (const auto& r : h->results())
                        f << r.solver_name         << ","
                          << r.backend_name        << ","
                          << r.grid_nx             << ","
                          << r.grid_ny             << ","
                          << r.iterations          << ","
                          << std::scientific << std::setprecision(6)
                          << r.final_residual      << ","
                          << r.normalized_residual << ","
                          << std::fixed << std::setprecision(4)
                          << r.wall_time_ms        << ","
                          << (r.converged ? "true" : "false") << ","
                          << r.fsm_state           << "\n";
                std::cout << "[IO] " << fname << "\n";
            }
            state = BenchmarkState::DONE;
            break;

        case BenchmarkState::DONE:
            break;
        }
    }

    std::cout << "\n=== Benchmark Complete ===\n";
}
