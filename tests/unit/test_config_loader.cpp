#include <gtest/gtest.h>
#include "core/SimulationParams.hpp"
#include "io/ConfigLoader.hpp"
#include <fstream>

using namespace physi_sim;

TEST(ConfigTest, ManualParamCreation) 
{
    // TDD: Testing the 'Contract' directly with dummy data
    core::SimulationParams params
    {
        .solver_type    = "random",
        .max_iterations = -1,
        .initial_temp   = -1.0
    };

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

TEST(ConfigTest, LoadsBenchmarkSection)
{
    // Write a config with the benchmark section
    std::string test_filename = "test_benchmark_config.json";
    std::ofstream out(test_filename);
    out << R"({
        "thermal": { "initial_temp": 25.5 },
        "solver":  { "max_iter": 1000, "type": "Jacobi" },
        "benchmark": {
            "grid_sizes": [50, 100, 200, 500],
            "tolerance":  1e-4,
            "run_gpu":    true,
            "output_dir": "/tmp"
        }
    })";
    out.close();

    auto params = io::ConfigLoader::load_json(test_filename);

    // existing fields still correct
    EXPECT_DOUBLE_EQ(params.initial_temp,   25.5);
    EXPECT_EQ(params.max_iterations,        1000);
    EXPECT_EQ(params.solver_type,           "Jacobi");

    // new benchmark fields
    ASSERT_EQ(params.grid_sizes.size(), 4u);
    EXPECT_EQ(params.grid_sizes[0],  50);
    EXPECT_EQ(params.grid_sizes[1], 100);
    EXPECT_EQ(params.grid_sizes[2], 200);
    EXPECT_EQ(params.grid_sizes[3], 500);
    EXPECT_DOUBLE_EQ(params.tolerance,  1e-4);
    EXPECT_TRUE(params.run_gpu);
    EXPECT_EQ(params.output_dir, "/tmp");
}

TEST(ConfigTest, MissingBenchmarkSectionUsesDefaults)
{
    // Config without benchmark section — must not throw, must use defaults
    std::string test_filename = "test_no_benchmark_config.json";
    std::ofstream out(test_filename);
    out << R"({
        "thermal": { "initial_temp": 25.5 },
        "solver":  { "max_iter": 1000, "type": "Jacobi" }
    })";
    out.close();

    auto params = io::ConfigLoader::load_json(test_filename);

    // defaults from SimulationParams
    ASSERT_EQ(params.grid_sizes.size(), 2u);   // {100, 500}
    EXPECT_EQ(params.grid_sizes[0], 100);
    EXPECT_EQ(params.grid_sizes[1], 500);
    EXPECT_DOUBLE_EQ(params.tolerance, 1e-7);
    EXPECT_TRUE(params.run_gpu);
    EXPECT_EQ(params.output_dir, ".");
}
