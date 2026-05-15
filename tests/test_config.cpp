#include <gtest/gtest.h>
#include "core/SimulationParams.hpp"
#include "io/ConfigLoader.hpp"
#include <fstream>

using namespace physi_sim;

TEST(ConfigTest, ManualParamCreation) {
    // TDD: Testing the 'Contract' directly with dummy data
    core::SimulationParams params{-1.0, -1, "random"};

    EXPECT_EQ(params.initial_temp, -1.0);
    EXPECT_EQ(params.max_iterations, -1);
    EXPECT_EQ(params.solver_type, "random");
    
    
    params.initial_temp = 100.0;
    params.max_iterations = 500;
    params.solver_type = "Thermal";

    EXPECT_EQ(params.initial_temp, 100.0);
    EXPECT_EQ(params.max_iterations, 500);
    EXPECT_EQ(params.solver_type, "Thermal");
}

TEST(ConfigTest, LoadFromJsonInterface) 
{
    // 1. Create a temporary config file for the test
    std::string test_filename = "test_config.json";
    std::ofstream out(test_filename);
    out << R"({
        "thermal": { "initial_temp": 25.5 },
        "solver": { "max_iter": 1000, "type": "Jacobi" }
    })";
    out.close();

    // 2. Act: Call your loader
    auto params = io::ConfigLoader::load_json(test_filename);

    // 3. Assert: Verify the data was mapped correctly
    EXPECT_DOUBLE_EQ(params.initial_temp, 25.5);
    EXPECT_EQ(params.max_iterations, 1000);
    EXPECT_EQ(params.solver_type, "Jacobi");
}

