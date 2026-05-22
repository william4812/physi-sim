#include <gtest/gtest.h>
#include "solver/ISolver.hpp"
#include "solver/SolverFactory.hpp"
//#include "solver/ProfilingHarness.hpp"
#include "core/Grid2D.hpp"

using namespace physi_sim::solver;
using namespace physi_sim::core;

class SolverFactoryTest : public ::testing::Test 
{
protected:
    void SetUp() override {
        grid = std::make_unique<Grid2D>(6, 6);
    }
    std::unique_ptr<Grid2D> grid;
};

TEST_F(SolverFactoryTest, CreatesJacobiCPU) 
{
    auto solver = SolverFactory::create(SolverType::JACOBI, HardwareBackend::CPU);
    ASSERT_NE(solver, nullptr);
    EXPECT_EQ(solver->name(), "JacobiCPU");
}

#ifdef PHYSI_SIM_CUDA_ENABLED
TEST_F(SolverFactoryTest, CreatesJacobiGPU) 
{
    // JACOBI+CUDA is now implemented — must return a valid solver
    auto solver = SolverFactory::create(SolverType::JACOBI, HardwareBackend::CUDA);
    ASSERT_NE(solver, nullptr);
    EXPECT_EQ(solver->name(), "JacobiGPU");
}
#endif

TEST_F(SolverFactoryTest, CreatesTDMACPU) 
{
    auto solver = SolverFactory::create(SolverType::TDMA, HardwareBackend::CPU);
    ASSERT_NE(solver, nullptr);
    EXPECT_EQ(solver->name(), "TDMACPU");
}

TEST_F(SolverFactoryTest, UnknownBackendThrows) 
{
    EXPECT_THROW(
        SolverFactory::create(SolverType::TDMA, HardwareBackend::CUDA),
        std::invalid_argument
    ); 
}

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
    EXPECT_THROW(SolverFactory::parseSolverType("foobar"), std::invalid_argument);
}

TEST_F(SolverFactoryTest, StepDoesNotThrow) 
{
    auto solver = SolverFactory::create(SolverType::JACOBI, HardwareBackend::CPU);
    EXPECT_NO_THROW(solver->step(*grid));
}

TEST_F(SolverFactoryTest, ResidualIsNonNegative) 
{
    auto solver = SolverFactory::create(SolverType::JACOBI, HardwareBackend::CPU);
    solver->step(*grid);
    EXPECT_GE(solver->residual(), 0.0);
}

TEST_F(SolverFactoryTest, TDMAResidualDecreasesOverIterations) 
{
    auto solver = SolverFactory::create(SolverType::TDMA, HardwareBackend::CPU);
    (*grid)(2, 2) = 1.0;
    (*grid)(3, 3) = 0.5;
    solver->step(*grid);
    const double r0 = solver->residual();
    for (int i = 0; i < 50; ++i) solver->step(*grid);
    const double r50 = solver->residual();
    EXPECT_LT(r50, r0)
        << "TDMA not converging: r0=" << r0 << " r50=" << r50;
}

