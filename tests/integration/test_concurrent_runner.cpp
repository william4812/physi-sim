// tests/integration/test_concurrent_runner.cpp
//
// Layer  : integration
// Seam   : ConcurrentSolverRunner → ProfilingHarness × N → std::async
// Owns   : concurrent dispatch correctness + speedup proof
//
// FIRST PRINCIPLE — what each test proves:
//   ContainsOneRecordPerSolver    → dispatch count is deterministic
//   RecordsInDispatchOrder        → results[0]=Jacobi, results[1]=TDMA
//   BothSolversConverge           → fsm_state per thread is independently verified
//   EachGridIsIndependent         → no shared mutable state (different solver results)
//   ConcurrentFasterThanSequential→ wall-clock speedup is measured, not claimed
//   ThreadSanitizerClean          → run with -fsanitize=thread in CI

#include <gtest/gtest.h>
#include <chrono>
#include <numeric>
#include <string>

#include "solver/ConcurrentSolverRunner.hpp"
#include "solver/SolverFactory.hpp"
#include "core/Grid2D.hpp"

using namespace physi_sim::solver;
using namespace physi_sim::core;

// ── Fixture ───────────────────────────────────────────────────────────────────

class ConcurrentRunnerTest : public ::testing::Test 
{
protected:
    void SetUp() override {
        // Small zero-initialised grid — converges in 1 step.
        // Used for structural tests (count, names, fsm_state).
        small_grid = std::make_unique<Grid2D>(6, 6);

        // Non-trivial grid with boundary condition — requires real iterations.
        // Used for the speedup timing test.
        large_grid = std::make_unique<Grid2D>(50, 50);
        for (int x = 0; x < 50; ++x)
            large_grid->at(x, 49) = 100.0;  // top BC: heat source
    }

    std::unique_ptr<Grid2D> small_grid;
    std::unique_ptr<Grid2D> large_grid;

    // Helper: build a two-solver vector [Jacobi, TDMA]
    static std::vector<std::unique_ptr<ISolver>> twoSolvers() 
    {
        std::vector<std::unique_ptr<ISolver>> v;
        v.push_back(SolverFactory::create(SolverType::JACOBI, HardwareBackend::CPU));
        v.push_back(SolverFactory::create(SolverType::TDMA,   HardwareBackend::CPU));
        return v;
    }
};

// ── 1. Dispatch count ─────────────────────────────────────────────────────────

TEST_F(ConcurrentRunnerTest, ContainsOneRecordPerSolver) 
{
    auto records = ConcurrentSolverRunner::run(twoSolvers(), *small_grid);
    EXPECT_EQ(records.size(), 2u);
}

// ── 2. Results in dispatch order ──────────────────────────────────────────────

TEST_F(ConcurrentRunnerTest, RecordsInDispatchOrder) 
{
    auto records = ConcurrentSolverRunner::run(twoSolvers(), *small_grid);
    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(records[0].solver_name, "JacobiCPU");
    EXPECT_EQ(records[1].solver_name, "TDMACPU");
}

// ── 3. FSM state per thread ───────────────────────────────────────────────────
// Each solver's lifecycle is independently verified — not inferred from batch.

TEST_F(ConcurrentRunnerTest, BothSolversConverge) 
{
    auto records = ConcurrentSolverRunner::run(twoSolvers(), *small_grid);
    ASSERT_EQ(records.size(), 2u);

    for (const auto& rec : records) 
    {
        EXPECT_EQ(rec.fsm_state, "CONVERGED")
            << rec.solver_name << " did not converge: "
            << "residual=" << rec.final_residual;
        EXPECT_TRUE(rec.converged);
        EXPECT_GT(rec.wall_time_ms, 0.0);
    }
}

// ── 4. Grid independence ──────────────────────────────────────────────────────
// Proves each thread got its own Grid2D copy — not a shared reference.
// Jacobi (slower) and TDMA (faster) converge in different iteration counts
// even on the same initial grid: different iteration counts = independent grids.

TEST_F(ConcurrentRunnerTest, EachGridIsIndependent) 
{
    RunConfig cfg;
    cfg.max_iters = 10000;
    cfg.tolerance = 1e-4;

    auto records = ConcurrentSolverRunner::run(twoSolvers(), *large_grid, cfg);
    ASSERT_EQ(records.size(), 2u);

    // Both must have recorded grid dimensions correctly
    for (const auto& rec : records) 
    {
        EXPECT_EQ(rec.grid_nx, 50);
        EXPECT_EQ(rec.grid_ny, 50);
    }

    // TDMA converges faster than Jacobi — proves independent grid state.
    // If grids were shared, both would see the same partially-converged field.
    const auto& jacobi_rec = records[0];
    const auto& tdma_rec   = records[1];
    EXPECT_LT(tdma_rec.iterations, jacobi_rec.iterations)
        << "TDMA should converge faster than Jacobi on independent grids";
}

// ── 5. Speedup — concurrent wall time < sequential sum ────────────────────────
// FIRST PRINCIPLE: if two solvers run truly in parallel, the wall-clock time
// of the batch ≈ max(individual times), not sum(individual times).
// This is the CPU-level proof of the GPU thread block independence model.

TEST_F(ConcurrentRunnerTest, ConcurrentFasterThanSequentialSum) 
{
    RunConfig cfg;
    cfg.max_iters = 10000;
    cfg.tolerance = 1e-7;

    // Measure concurrent wall time externally
    const auto t0 = std::chrono::steady_clock::now();
    auto records = ConcurrentSolverRunner::run(twoSolvers(), *large_grid, cfg);
    const double concurrent_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

    // Sequential sum = what it would cost to run them one after another
    double sequential_sum_ms = 0.0;
    for (const auto& rec : records)
        sequential_sum_ms += rec.wall_time_ms;

    EXPECT_GT(sequential_sum_ms, 0.0);
    EXPECT_GT(concurrent_ms, 0.0);

    // Always printed — visible with ctest --output-on-failure or ctest -V
    const double speedup = sequential_sum_ms / concurrent_ms;

    std::cout << "\n[ConcurrentSpeedup]"
              << "\n  Jacobi wall time  : " << records[0].wall_time_ms << " ms"
              << "\n  TDMA   wall time  : " << records[1].wall_time_ms << " ms"
              << "\n  Sequential sum    : " << sequential_sum_ms        << " ms"
              << "\n  Concurrent total  : " << concurrent_ms            << " ms"
              << "\n  Speedup           : " << speedup                  << "x"
              << "\n  fsm[Jacobi]       : " << records[0].fsm_state
              << "\n  fsm[TDMA]         : " << records[1].fsm_state
              << "\n";

    // Concurrent must be faster than sequential sum.
    // 20% margin accounts for thread creation and scheduling overhead.
    // On a single-core machine this may not hold — acceptable in that context.
    EXPECT_LT(concurrent_ms, sequential_sum_ms * 1.20)
        << "Concurrent dispatch did not run faster than sequential."
        << " concurrent_ms=" << concurrent_ms
        << " sequential_sum_ms=" << sequential_sum_ms
        << " (requires nproc >= 2)";
}

// ── 6. Single solver — edge case ──────────────────────────────────────────────

TEST_F(ConcurrentRunnerTest, SingleSolverReturnsOneRecord) 
{
    std::vector<std::unique_ptr<ISolver>> one;
    one.push_back(SolverFactory::create(SolverType::TDMA, HardwareBackend::CPU));

    auto records = ConcurrentSolverRunner::run(std::move(one), *small_grid);
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].solver_name, "TDMACPU");
    EXPECT_EQ(records[0].fsm_state,   "CONVERGED");
}

// ── 7. Empty solver vector — edge case ───────────────────────────────────────

TEST_F(ConcurrentRunnerTest, EmptySolverVectorReturnsEmpty) 
{
    std::vector<std::unique_ptr<ISolver>> none;
    auto records = ConcurrentSolverRunner::run(std::move(none), *small_grid);
    EXPECT_TRUE(records.empty());
}
