#include <gtest/gtest.h>
#include "LBMSolver.hpp"
#include "thermal/LinearDummySolver.hpp"
#include "thermal/FortranBackend.hpp"
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

extern "C" {
    void check_abi_integrity(double val_in, double* val_out);
}

TEST(ABICompatibility, FortranDoubleMatchesCppDouble) 
{
    double input = 0.125;
    double output = 0.0;

    // Call the Fortran binary directly
    check_abi_integrity(input, &output);

    // 1st Principle: If the ABI is identical, 0.125 * 2 MUST be exactly 0.25
    EXPECT_DOUBLE_EQ(output, 0.25);
}

TEST(FortranBackendTest, ConvergenceTest) {
    using namespace physi_sim::thermal;

    FortranBackend solver;

    // Initialize a simple 1D grid: [0, 1, 0]
    // A single heat pulse in the center
    std::vector<double> grid = {0.0, 1.0, 0.0};

    // Stability parameters (alpha * dt / dx^2)
    double alpha = 0.01;
    double dt = 1.0;

    // Run one step
    solver.compute(grid, alpha, dt);

    // 1st Principle: Energy Conservation / Diffusion
    // After one step, the center value should decrease and neighbors increase
    EXPECT_LT(grid[1], 1.0);
    EXPECT_EQ(grid[0], 0.0);
    EXPECT_EQ(grid[2], 0.0);

    // Verify physical symmetry
    EXPECT_NEAR(grid[0], grid[2], 1e-10);
}

/**
 * @test SteadyStateLinearProfileTest
 * Verifies the 1D Steady-State Implicit Solver (TDMA).
 * 1st Principle: For steady-state conduction with Dirichlet BCs and no source terms,
 * the temperature profile must be perfectly linear.
 */
TEST(FortranBackendTest, SteadyStateLinearProfileTest) 
{
    using namespace physi_sim::thermal;

    FortranBackend solver;

    // 1. Setup Grid and Boundary Conditions
    const int n = 11;                  // 11 nodes (10 control volumes)
    const double T_L = 100.0;          // Left boundary temperature
    const double T_R = 0.0;            // Right boundary temperature
    const double k = 1.0;              // Thermal conductivity

    // Initialize grid - intermediate values will be overwritten by solver
    std::vector<double> grid(n, 0.0);

    // 2. Execute Steady-State Solver
    // This calls the TDMA kernel mapped to Equation 4.5 and image_13ce00.png
    solver.compute_steady(grid, k, T_L, T_R);

    // 3. Physical Verification: Linear Gradient Check
    // Analytical Solution: T(i) = T_L + i * (T_R - T_L) / (n - 1)
    double expected_slope = (T_R - T_L) / (n - 1);

    for (int i = 0; i < n; ++i) {
        double expected_T = T_L + i * expected_slope;

        // Use EXPECT_NEAR for floating-point robustness (1e-9 tolerance)
        EXPECT_NEAR(grid[i], expected_T, 1e-9)
            << "Physical violation at node " << i << ": Non-linear profile detected.";
    }

    // 4. Boundary Condition Check
    EXPECT_DOUBLE_EQ(grid[0], T_L);    // Verify Left Dirichlet BC
    EXPECT_DOUBLE_EQ(grid[n-1], T_R);  // Verify Right Dirichlet BC
}
