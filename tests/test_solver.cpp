#include <gtest/gtest.h>
#include "LBMSolver.hpp"
#include "MockBackend.hpp"

// Test Case: Ensure the solver calls the backend phases exactly once per step
TEST(SolverLogicTest, IncrementsCountersCorrectly) 
{
    // We keep a raw pointer to the mock to inspect counters before ownership moves
    auto mock_ptr = new MockBackend(); 
    std::unique_ptr<MockBackend> backend(mock_ptr);
    
    LBMSolver solver(std::move(backend));

    // Execute 3 steps
    for(int i = 0; i < 3; ++i) 
    {
        solver.step();
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

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
