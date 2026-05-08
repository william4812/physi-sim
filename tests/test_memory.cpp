#include <gtest/gtest.h>
#include "core/Grid2D.hpp"

TEST(MemoryArchitectureTest, Flattened2DStrideIsCorrect) {
    int NX = 4;
    int NY = 3;
    // Create a flattened grid: Size 12
    physi_sim::core::Grid2D grid(NX, NY); 

    // Set a value at (x=2, y=1)
    // Index should be: (y * NX) + x => (1 * 4) + 2 = 6
    grid.at(2, 1) = 42.0;

    // Verify the underlying memory layout
    EXPECT_EQ(grid.get_raw_data()[6], 42.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
