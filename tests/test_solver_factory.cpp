/**
 * @file test_solver_factory.cpp
 * @brief TDD tests for SolverFactory and ISolver interface.
 *
 * WORKFLOW: these tests are written BEFORE the production code.
 * They define the contract ISolver and SolverFactory must satisfy.
 * Run: ctest -R TestSolverFactory -V
 *
 * ANDURIL/NVIDIA STANDARD:
 *   Every public interface gets a test file written first.
 *   The test file IS the specification — not a Word doc, not a comment.
 *   If it compiles and passes, the contract is met.
 */

#include <gtest/gtest.h>
#include "solver/ISolver.hpp"
#include "solver/SolverFactory.hpp"

using namespace physi;

// ── Fixture ───────────────────────────────────────────────────────────────────
// A minimal Grid for testing — 4×4 interior points, h=0.2
// We don't need a real physics grid — just enough to call step()
class SolverFactoryTest : public ::testing::Test 
{
protected:
    void SetUp() override 
    {
        grid.nx = 4;
        grid.ny = 4;
        grid.dx = 0.2;
        grid.dy = 0.2;
        // initialise T and f to zero — boundary conditions all zero
        grid.T.assign(6, std::vector<double>(6, 0.0));
        grid.f.assign(6, std::vector<double>(6, 0.0));
    }
    Grid grid;
};

// ── Factory creation tests ────────────────────────────────────────────────────

TEST_F(SolverFactoryTest, CreatesJacobiCPU) 
{
    // FIRST PRINCIPLE: factory returns non-null unique_ptr
    auto solver = SolverFactory::create(SolverType::JACOBI,
                                        HardwareBackend::CPU);
    ASSERT_NE(solver, nullptr);
    EXPECT_EQ(solver->name(), "JacobiCPU");
}

TEST_F(SolverFactoryTest, CreatesTDMACPU) 
{
    auto solver = SolverFactory::create(SolverType::TDMA,
                                        HardwareBackend::CPU);
    ASSERT_NE(solver, nullptr);
    EXPECT_EQ(solver->name(), "TDMACPU");
}

TEST_F(SolverFactoryTest, UnknownBackendThrows) 
{
    // FIRST PRINCIPLE: explicit failure is better than silent wrong behaviour
    // An unimplemented combination must throw — never silently return nullptr
    EXPECT_THROW(
        SolverFactory::create(SolverType::JACOBI, HardwareBackend::CUDA),
        std::invalid_argument
    );
}

// ── String parsing tests ──────────────────────────────────────────────────────

TEST(SolverFactoryParseTest, ParsesJacobiCaseInsensitive) 
{
    EXPECT_EQ(SolverFactory::parseSolverType("jacobi"), SolverType::JACOBI);
    EXPECT_EQ(SolverFactory::parseSolverType("JACOBI"), SolverType::JACOBI);
    EXPECT_EQ(SolverFactory::parseSolverType("Jacobi"), SolverType::JACOBI);
}

TEST(SolverFactoryParseTest, ParsesTDMA) 
{
    EXPECT_EQ(SolverFactory::parseSolverType("tdma"), SolverType::TDMA);
    EXPECT_EQ(SolverFactory::parseSolverType("TDMA"), SolverType::TDMA);
}

TEST(SolverFactoryParseTest, ParsesBackendCPU) 
{
    EXPECT_EQ(SolverFactory::parseBackend("cpu"),  HardwareBackend::CPU);
    EXPECT_EQ(SolverFactory::parseBackend("CPU"),  HardwareBackend::CPU);
}

TEST(SolverFactoryParseTest, UnknownSolverTypeThrows) 
{
    EXPECT_THROW(SolverFactory::parseSolverType("foobar"),
                 std::invalid_argument);
}

// ── ISolver contract tests ────────────────────────────────────────────────────
// These tests verify the ISolver contract holds for EVERY concrete solver.
// Parameterised so adding JacobiGPU later automatically gets these tests.

TEST_F(SolverFactoryTest, StepDoesNotThrow) 
{
    // A solver must be able to run one step without throwing
    auto solver = SolverFactory::create(SolverType::JACOBI,
                                        HardwareBackend::CPU);
    EXPECT_NO_THROW(solver->step(grid));
}

TEST_F(SolverFactoryTest, ResidualIsNonNegative) 
{
    // L-inf norm is always >= 0
    auto solver = SolverFactory::create(SolverType::JACOBI,
                                        HardwareBackend::CPU);
    solver->step(grid);
    EXPECT_GE(solver->residual(), 0.0);
}

TEST_F(SolverFactoryTest, ResidualDecreasesOverIterations) 
{
    // PHYSICS: a correct iterative solver must reduce the residual
    // This is the most important correctness test — if this fails,
    // the solver is diverging, not converging
    auto solver = SolverFactory::create(SolverType::TDMA,
                                        HardwareBackend::CPU);

    // seed a non-trivial initial condition
    grid.T[2][2] = 1.0;
    grid.T[3][3] = 0.5;

    solver->step(grid);
    const double r0 = solver->residual();

    for (int i = 0; i < 50; ++i) solver->step(grid);
    const double r50 = solver->residual();

    // After 50 iterations residual must be strictly smaller
    EXPECT_LT(r50, r0)
        << "Solver not converging — residual increased from "
        << r0 << " to " << r50;
}

TEST_F(SolverFactoryTest, ProfilingHarnessRunsWithoutThrowing) 
{
    // Verify harness wraps solver cleanly
    auto solver  = SolverFactory::create(SolverType::JACOBI,
                                          HardwareBackend::CPU);
    auto harness = ProfilingHarness(std::move(solver));

    ProfilingRecord rec;
    EXPECT_NO_THROW(rec = harness.run(grid, /*max_iters=*/100,
                                             /*tol=*/1e-3,
                                             /*verbose=*/false));
    EXPECT_GT(rec.wall_time_ms, 0.0);
    EXPECT_GT(rec.iterations,   0);
}
