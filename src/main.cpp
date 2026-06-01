#include "MockBackend.hpp"
#include "thermal/LinearDummySolver.hpp"
#include "thermal/FortranBackend.hpp"
#include "io/VTKWriter.hpp"
#include "io/ConfigLoader.hpp"
#include "solver/SolverFactory.hpp"      
#include "solver/ProfilingHarness.hpp"
#include "core/Grid2D.hpp"
#ifdef PHYSI_SIM_CUDA_ENABLED
#include "solver/CudaJacobiSolver.hpp"
#endif
#include "solver/JacobiCPU.hpp"
#include "solver/TDMACPU.hpp"       // direct use in comparison runners
#include "solver/ISolver.hpp"       // solve_to_tol() takes ISolver&
#include <fstream>                  // CSV writing
#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>

// forward declaration
void run_thermal_benchmark();
void run_solver_benchmark(const physi_sim::core::SimulationParams& params);
#ifdef PHYSI_SIM_CUDA_ENABLED
void run_comparison_exports(const physi_sim::core::SimulationParams& params);           // controlled-comparison field/timing exporter
#endif

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

#ifdef PHYSI_SIM_CUDA_ENABLED
        // Controlled-comparison ladder: writes cmp_*.vtk + cmp_timing.csv,
        // consumed by the Python comparison figures.
        run_comparison_exports(params);
#endif
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


#ifdef PHYSI_SIM_CUDA_ENABLED
// ═════════════════════════════════════════════════════════════════════════════
// Controlled-comparison exporter — modular runners + thin orchestrator.
//
// Each runner owns ONE experiment cell: it runs one solver variant to the same
// tolerance, on the same grid + boundary condition, writes its converged field
// to VTK, and returns a structured result. run_comparison_exports() composes
// whichever variants you want — comment a line out to skip it.
//
// Each figure changes exactly ONE variable:
//   Fig 1  algorithm      : JacobiCPU      vs TDMACPU         (CPU held)
//   Fig 2  backend        : JacobiCPU      vs JacobiGPU step  (algorithm held)
//   Fig 3  data residency : JacobiGPU step vs JacobiGPU VRAM  (backend held)
//   Fig 4  algorithm/GPU  : (future) TDMA-GPU-VRAM vs Jacobi-GPU-VRAM
//
// Absolute L-inf tolerance (matches the unit-test field comparisons) — NOT the
// normalized criterion the grid-sweep above uses. Do not mix the numbers.
// ═════════════════════════════════════════════════════════════════════════════

struct CompareConfig {
    int    N   = 100;
    double tol = 5e-4;     // absolute L-inf
    int    cap = 20000;    // hard iteration ceiling
};

struct RunResult {
    std::string variant;     // "JacobiCPU", "JacobiGPU_vram", ...
    std::string backend;     // "cpu" | "cuda"
    std::string residency;   // "host" | "per_iter" | "resident"
    int         iterations = 0;
    double      wall_ms     = 0.0;
    std::string vtk_path;
};

// ── Shared helpers — the only common logic, factored once ─────────────────────
static physi_sim::core::Grid2D fresh_grid(const CompareConfig& cfg)
{
    physi_sim::core::Grid2D g(cfg.N, cfg.N);
    for (int x = 0; x < cfg.N; ++x) g(x, cfg.N - 1) = 100.0;   // T_top = 100
    return g;
}

// Drive any ISolver to tolerance via step(); time physics only; return iters.
static int solve_to_tol(physi_sim::solver::ISolver& s,
                        physi_sim::core::Grid2D& g,
                        const CompareConfig& cfg, double& ms)
{
    int it = 0;
    const auto t0 = std::chrono::high_resolution_clock::now();
    
    do 
    { 
        s.step(g); ++it; 
    } while (s.residual() > cfg.tol && it < cfg.cap);

    ms = std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - t0).count();
    return it;
}

// ── One function per experiment cell ──────────────────────────────────────────
RunResult runJacobiCPU(const CompareConfig& cfg)
{
    using namespace physi_sim;
    auto g = fresh_grid(cfg); 
    double ms; 
    solver::JacobiCPU s;
    int it = solve_to_tol(s, g, cfg, ms);
    const std::string path = "cmp_cpu_jacobi_" + std::to_string(cfg.N) + ".vtk";
    
    io::VTKWriter().write_2d(g.data(), cfg.N, cfg.N, path);
    return RunResult{ .variant="JacobiCPU", .backend="cpu", .residency="host",
                      .iterations=it, .wall_ms=ms, .vtk_path=path };
}

RunResult runTDMACPU(const CompareConfig& cfg)
{
    using namespace physi_sim;
    auto g = fresh_grid(cfg); 
    double ms; 
    solver::TDMACPU s;
    int it = solve_to_tol(s, g, cfg, ms);
    const std::string path = "cmp_cpu_tdma_"+ std::to_string(cfg.N) + ".vtk";
    
    io::VTKWriter().write_2d(g.data(), cfg.N, cfg.N, path);
    return RunResult{ .variant="TDMACPU", .backend="cpu", .residency="host",
                      .iterations=it, .wall_ms=ms, .vtk_path=path };
}

RunResult runJacobiGPUNoVram(const CompareConfig& cfg)  // step-based: PCIe every iteration
{
    using namespace physi_sim;
    auto g = fresh_grid(cfg); double ms; solver::CudaJacobiSolver s;
    int it = solve_to_tol(s, g, cfg, ms);               // step() = H2D + kernel + D2H per iter
    const std::string path = "cmp_gpu_jacobi_novram_" + std::to_string(cfg.N) + ".vtk";
    
    io::VTKWriter().write_2d(g.data(), cfg.N, cfg.N, path);
    return RunResult{ .variant="JacobiGPU_NoVram", .backend="cuda", .residency="per_iter",
                      .iterations=it, .wall_ms=ms, .vtk_path=path };
}

RunResult runJacobiGPUVram(const CompareConfig& cfg)    // VRAM-resident
{
    using namespace physi_sim;
    auto g = fresh_grid(cfg); 
    double ms; 
    solver::CudaJacobiSolver s;
    const auto t0 = std::chrono::high_resolution_clock::now();
    s.upload(g);
    s.solve_vram(cfg.cap, cfg.tol);
    s.download(g);
    ms = std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - t0).count();
    const std::string path = "cmp_gpu_jacobi_vram_" + std::to_string(cfg.N) + ".vtk";
    
    io::VTKWriter().write_2d(g.data(), cfg.N, cfg.N, path);
    // NOTE: if your header names this accessor differently
    // (e.g. vram_iteration_count()), change the call below to match.
    return RunResult{ .variant="JacobiGPU_Vram", .backend="cuda", .residency="resident",
                      .iterations=s.get_vram_iterations(), .wall_ms=ms, .vtk_path=path };
}

// ── Orchestrator — call some or all; comment a line out to skip a variant ─────
void run_comparison_exports(const physi_sim::core::SimulationParams& params)
{
    // NOTE: main.cpp is compiled by the host C++ compiler, which has no CUDA
    // include path — so we can't call cudaGetDeviceCount() here. If CUDA is
    // enabled but no device is present, the first cudaMalloc inside
    // CudaJacobiSolver throws std::runtime_error, which main()'s try/catch
    // already handles (same as run_solver_benchmark above).
    CompareConfig cfg;     // one grid / tol / BC for ALL variants

    if (params.compare_grid > 0)             // honor JSON only if present/valid
        cfg.N = params.compare_grid;

    const std::vector<RunResult> results = {
        runJacobiCPU(cfg),       // Fig 1 + Fig 2 CPU side
        runTDMACPU(cfg),         // Fig 1
        runJacobiGPUNoVram(cfg), // Fig 2 + Fig 3 (no-VRAM, PCIe per iter)
        runJacobiGPUVram(cfg),   // Fig 3 (VRAM-resident)
        // runTDMAGPUVram(cfg),  // ← future (Phase 5): add the runner, uncomment here
    };

    // Single timing table — its only job is to serialize the results.
    std::ofstream csv("cmp_timing.csv");
    csv << "variant,backend,residency,grid_n,iterations,wall_time_ms\n"
        << std::fixed << std::setprecision(4);
    for (const auto& r : results)
        csv << r.variant << "," << r.backend << "," << r.residency << ","
            << cfg.N << "," <<  r.iterations << "," << r.wall_ms << "\n";

    std::cout << "\n=== Comparison exports — " << results.size()
              << " variants ===\n";
    for (const auto& r : results)
        std::cout << "  " << r.variant << " [" << cfg.N << "x" << cfg.N << "]: "
          << r.iterations << " it / " << r.wall_ms << " ms  → " << r.vtk_path << "\n";
        //std::cout << "  " << r.variant << ": " << r.iterations
        //          << " it / " << r.wall_ms << " ms  → " << r.vtk_path << "\n";
    std::cout << "[IO] cmp_timing.csv\n";
}
#endif // PHYSI_SIM_CUDA_ENABLED
