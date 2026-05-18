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
    void SetUp() override {
        grid = std::make_unique<Grid2D>(6, 6);
    }
    std::unique_ptr<Grid2D> grid;
};

TEST_F(ProfilingHarnessTest, FSMStateIsConvergedAfterRun) 
{
    auto solver  = SolverFactory::create(SolverType::TDMA, HardwareBackend::CPU);
    auto harness = ProfilingHarness(std::move(solver));
    auto rec     = harness.run(*grid, 200, 1e-6, false);

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
