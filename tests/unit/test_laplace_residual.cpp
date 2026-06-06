/**
 * @file test_laplace_residual.cpp
 * @brief Proves the unified equation-residual function (core::laplace_residual_linf)
 *        is correct, and connects it to JacobiCPU WITHOUT modifying the solver.
 *
 * Three tests, increasing in strength:
 *   1. KnownFieldGivesKnownResidual    — pure-function correctness, no solver.
 *   2. EqualsFourTimesJacobiIncrement  — ties the function to JacobiCPU via the
 *                                        exact identity  equation = 4 × increment.
 *   3. ConvergedJacobiFieldHasSmallResidual — end-to-end: a field JacobiCPU
 *                                        declares "converged" really does solve
 *                                        the equation, by the unified yardstick.
 *
 * Guards: include/core/LaplaceResidual.hpp (the measure) and, via test 3,
 * JacobiCPU.cpp + laplace_2d_jacobi (the iteration that must satisfy it).
 */
#include <gtest/gtest.h>
#include "core/LaplaceResidual.hpp"
#include "core/Grid2D.hpp"
#include "solver/JacobiCPU.hpp"
#include <cmath>

using namespace physi_sim;

namespace {
constexpr int    N   = 60;
constexpr double TOL = 5e-4;     // increment tolerance JacobiCPU stops at
constexpr int    CAP = 20000;

core::Grid2D fresh() {
    core::Grid2D g(N, N);
    for (int x = 0; x < N; ++x) g(x, N - 1) = 100.0;   // T_top = 100, rest 0
    return g;
}
} // namespace

// 1. Pure-function correctness — no solver involved.
// Build a field by hand whose residual we can compute on paper, and check it.
// All-zero grid with a single interior node set to 1.0:
//   - at that node:        |0+0+0+0 - 4·1| = 4
//   - at each neighbour:   |1+0+0+0 - 4·0| = 1
//   => the worst interior node is 4.0, exactly.
TEST(LaplaceResidual, KnownFieldGivesKnownResidual) {
    core::Grid2D g(N, N);                 // all zeros
    g(N / 2, N / 2) = 1.0;                // one interior spike
    EXPECT_DOUBLE_EQ(core::laplace_residual_linf(g), 4.0)
        << "stencil, abs, or max is wrong — this is hand-computable";
}

// 2. The exact identity that makes this the *equation* residual.
// For a Jacobi step, increment_ij = ¼·(equation residual_ij) pointwise, so the
// L-inf equation residual of the PRE-step field must equal 4 × the increment
// that step reports. We verify on the first step from `fresh()`:
//   node just below the hot row has increment ¼·(100) = 25, residual = 100.
TEST(LaplaceResidual, EqualsFourTimesJacobiIncrement) {
    core::Grid2D g = fresh();

    // Equation residual of the field BEFORE the step.
    const double eq_before = core::laplace_residual_linf(g);

    // One Jacobi step; its residual() is the increment for that same field.
    solver::JacobiCPU s;
    s.step(g);
    const double increment = s.residual();

    EXPECT_NEAR(eq_before, 4.0 * increment, 1e-9 + 1e-9 * std::abs(eq_before))
        << "equation residual must be exactly 4× the Jacobi increment — if not, "
           "laplace_residual_linf is not computing the true equation residual";
}

// 3. End-to-end on JacobiCPU. Converge using the solver's own (increment)
// criterion, then confirm the unified equation residual agrees the field is
// solved. Since increment < TOL and equation = 4 × increment, the equation
// residual must be below 4·TOL.
TEST(LaplaceResidual, ConvergedJacobiFieldHasSmallResidual) {
    core::Grid2D g = fresh();
    solver::JacobiCPU s;

    int it = 0;
    do { s.step(g); } while (s.residual() > TOL && ++it < CAP);

    EXPECT_LT(core::laplace_residual_linf(g), 4.0 * TOL)
        << "JacobiCPU reported convergence, but the equation residual is large — "
           "the field does not actually solve the Laplace equation";
}
