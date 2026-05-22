#include <gtest/gtest.h>   // TEST_F, EXPECT_*, ASSERT_*
#include <fstream>          // std::ifstream — reading CSV back in writeCSV tests
#include <string>           // std::string, std::getline

#include "solver/ProfilingHarness.hpp"  // class under test — ProfilingHarness, ProfilingRecord
#include "solver/SolverFactory.hpp"     // SolverFactory::create(), SolverType, HardwareBackend
#include "solver/SolverFSM.hpp"         // SolverState enum — for fsmState() assertions
#include "core/Grid2D.hpp"              // Grid2D — used in fixture SetUp()

using namespace physi_sim::solver;
using namespace physi_sim::core;

class ProfilingHarnessTest : public ::testing::Test 
{
protected:
    void SetUp() override 
    {
        grid = std::make_unique<Grid2D>(6, 6);
    }
    std::unique_ptr<Grid2D> grid;
};

TEST_F(ProfilingHarnessTest, FSMStateIsConvergedAfterRun) 
{
    auto solver  = SolverFactory::create(SolverType::TDMA, HardwareBackend::CPU);
    auto harness = ProfilingHarness(std::move(solver));
    auto rec     = harness.run(*grid, 200, 1e-4, false);

    EXPECT_EQ(rec.fsm_state, "CONVERGED");
    EXPECT_EQ(rec.converged, true);
    EXPECT_EQ(harness.fsmState(), SolverState::IDLE);
}

TEST_F(ProfilingHarnessTest, CSVContainsFSMStateColumn) 
{
    auto solver  = SolverFactory::create(SolverType::JACOBI, HardwareBackend::CPU);
    auto harness = ProfilingHarness(std::move(solver));
    auto rec = harness.run(*grid, 50, 1e-4, false);

    harness.writeCSV("/tmp/test_fsm_csv.csv");

    std::ifstream f("/tmp/test_fsm_csv.csv");
    std::string header;
    std::getline(f, header);
    EXPECT_NE(header.find("fsm_state"), std::string::npos);
}

TEST_F(ProfilingHarnessTest, NormalizedResidualIsBetweenZeroAndOne)
{
    // normalized = res / residual_initial
    // After convergence on a well-posed problem it must be in (0, 1]
    auto solver  = SolverFactory::create(SolverType::JACOBI, HardwareBackend::CPU);
    auto harness = ProfilingHarness(std::move(solver));

    // Use a grid with a real boundary condition so residual_initial > 0
    Grid2D g(20, 20);
    for (int x = 0; x < 20; ++x) g(x, 19) = 100.0;

    auto rec = harness.run(g, 10000, 1e-4, false);

    EXPECT_GT(rec.normalized_residual, 0.0)
        << "normalized_residual must be positive after convergence";
    EXPECT_LE(rec.normalized_residual, 1.0)
        << "normalized_residual cannot exceed 1.0 — residual grew above initial";
}

TEST_F(ProfilingHarnessTest, CSVContainsNormalizedResidualColumn)
{
    auto solver  = SolverFactory::create(SolverType::JACOBI, HardwareBackend::CPU);
    auto harness = ProfilingHarness(std::move(solver));
    auto rec = harness.run(*grid, 50, 1e-4, false);

    harness.writeCSV("/tmp/test_normalized_csv.csv");

    std::ifstream f("/tmp/test_normalized_csv.csv");
    std::string header;
    std::getline(f, header);

    EXPECT_NE(header.find("normalized_residual"), std::string::npos)
        << "CSV header missing normalized_residual column";

    // Also verify column order: final_residual comes before normalized_residual
    EXPECT_LT(header.find("final_residual"), header.find("normalized_residual"))
        << "Column order wrong: final_residual must precede normalized_residual";
}
