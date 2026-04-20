#include <gtest/gtest.h>
#include "LBMSolver.hpp"
#include "thermal/LinearDummySolver.hpp"
#include "MockBackend.hpp"

// Test Case: Ensure the solver calls the backend phases exactly once per step
TEST(SolverLogicTest, IncrementsCountersCorrectly) 
{
    // We keep a raw pointer to the mock to inspect counters before ownership moves
    auto mock_ptr = new MockBackend(); 
    std::unique_ptr<MockBackend> backend(mock_ptr);
    
    LBMSolver solver(std::move(backend));

    double dx{1.0};
    double dt{1.0};
    // Execute 3 steps
    for(int i = 0; i < 3; ++i) 
    {
        solver.step(dt, dx);
    }

    // Verify the hardware was called the correct number of times
    EXPECT_EQ(mock_ptr->collision_count, 3);
    EXPECT_EQ(mock_ptr->stream_count, 3);
}

// Test Case: Data Integrity Check
TEST(DataIntegrityTest, FailsOnEmptyVector) 
{
    MockBackend backend;
    std::vector<double> empty_vec;
    
    // We expect the mock to handle the empty vector gracefully per your implementation
    // In a real L6 scenario, you might expect an exception here.
    EXPECT_NO_THROW(backend.syncToHost(empty_vec));
}

TEST(LinearDummySolverTest, ReturnsValidFieldSize) {
    // 1. Setup (Dependency Injection)
    auto backend = std::make_unique<MockBackend>();
    auto size{1};
    LinearDummySolver solver(std::move(backend),
            static_cast<size_t>(size));

    // 2. Action
    auto field = solver.getTemperatureField();

    // 3. Assert (This will pass with your current code)
    ASSERT_FALSE(field.empty());
    EXPECT_EQ(field.size(), 1);
    EXPECT_DOUBLE_EQ(field[0], 0.0);
}

TEST(LinearDummySolverTest, ReturnsLinearResult) {
    // 1. Setup (Dependency Injection)
    auto backend = std::make_unique<MockBackend>();
    auto size{100};
    LinearDummySolver solver(std::move(backend), 
            static_cast<size_t>(size));

    // 2. step increment the values of elements
    double dt{0.0};
    double dx{0.0};
    for (auto i = 0; i < size; ++i) 
    {
        solver.step(dx, dt);
    }

    // 2. Action
    auto field = solver.getTemperatureField();

    // 3. Assert (This will pass with your current code)
    ASSERT_FALSE(field.empty());
    EXPECT_EQ(field.size(), 100);
    EXPECT_NEAR(field[99], 100.0, 1e-6);
}
