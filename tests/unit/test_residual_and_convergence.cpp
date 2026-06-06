/**
 * @file test_residual_and_convergence.cpp
 * @brief Rigorous, CFD-grade verification in two parts.
 *
 *   PART A — atomic properties of the residual OPERATOR (no solver involved).
 *     The discrete 5-point Laplacian residual must obey analytic identities.
 *     Each test uses a field whose residual is known by hand / by calculus:
 *       A1 linear field      -> residual exactly 0   (operator must VANISH on
 *                               an exact discrete-harmonic solution)
 *       A2 T = x^2           -> residual exactly 2    (known discrete Laplacian)
 *       A3 shift by +c       -> residual unchanged    (Laplacian kills constants;
 *                               physically: invariant to reference temperature)
 *       A4 scale by alpha    -> residual scales |alpha| (operator linearity)
 *
 *   PART B — convergence-vs-iteration behaviour of a real solve (JacobiCPU),
 *     checked against what stationary-iteration theory REQUIRES:
 *       B1 the equation residual is monotonically non-increasing (contraction)
 *       B2 a tighter tolerance never converges in fewer iterations
 *       B3 the asymptotic per-iteration reduction matches the Jacobi spectral
 *          radius  rho = cos(pi/(N-1))  — the textbook convergence rate
 *       B4 equation residual = 4 x Jacobi increment at EVERY iteration
 *
 * Part B builds its OWN equation-residual history by calling
 * core::laplace_residual_linf(grid) after each step(), so it needs no solver
 * change and would work identically for any ISolver.
 */
#include <gtest/gtest.h>
#include "core/LaplaceResidual.hpp"
#include "core/Grid2D.hpp"
#include "solver/JacobiCPU.hpp"
#include <cmath>
#include <vector>

using namespace physi_sim;

namespace {
constexpr double PI = 3.14159265358979323846;   // local, portable (no M_PI dep)

// Standard problem: T_top = 100, all else 0, on an N x N grid.
core::Grid2D fresh(int N) {
    core::Grid2D g(N, N);
    for (int x = 0; x < N; ++x) g(x, N - 1) = 100.0;
    return g;
}

// Drive JacobiCPU and record the EQUATION residual after each step.
// Independent of how the solver defines its own (increment) residual.
std::vector<double> jacobi_equation_history(int N, int steps) {
    core::Grid2D g = fresh(N);
    solver::JacobiCPU s;
    std::vector<double> hist;
    hist.reserve(steps);
    for (int k = 0; k < steps; ++k) {
        s.step(g);
        hist.push_back(core::laplace_residual_linf(g));
    }
    return hist;
}
} // namespace

// ════════════════════════════════════════════════════════════════════════════
// PART A — atomic operator properties (pure function, exact analytic values)
// ════════════════════════════════════════════════════════════════════════════

// A1. A discrete-harmonic field must give ZERO residual.
// The 5-point stencil is exact for linear functions: for T = a + b*x + c*y it
// cancels to 0 at every node. A correct residual operator MUST vanish on an
// exact solution — this is the single most important property.
TEST(ResidualOperator, LinearFieldGivesZeroResidual) {
    const int N = 16;
    core::Grid2D g(N, N);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            g(i, j) = 5.0 + 2.0 * i + 3.0 * j;
    EXPECT_NEAR(core::laplace_residual_linf(g), 0.0, 1e-9)
        << "operator does not vanish on a linear (exact-harmonic) field";
}

// A2. A field with KNOWN curvature gives its known residual.
// For T = x^2 the discrete Laplacian is exactly 2 at every interior node:
//   (i+1)^2 + (i-1)^2 + i^2 + i^2 - 4 i^2 = 2.
TEST(ResidualOperator, QuadraticFieldGivesKnownResidual) {
    const int N = 16;
    core::Grid2D g(N, N);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            g(i, j) = static_cast<double>(i) * i;   // T = x^2
    EXPECT_NEAR(core::laplace_residual_linf(g), 2.0, 1e-9)
        << "operator does not reproduce the known Laplacian of x^2";
}

// A3. Adding a constant must not change the residual.
// The Laplacian annihilates constants (4c - 4c = 0). Physically: steady state
// is invariant to the reference temperature. Guards against any accidental
// absolute-temperature dependence.
TEST(ResidualOperator, ConstantShiftInvariance) {
    const int N = 24;
    core::Grid2D a = fresh(N);
    core::Grid2D b(N, N);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            b(i, j) = a(i, j) + 50.0;
    EXPECT_NEAR(core::laplace_residual_linf(a),
                core::laplace_residual_linf(b), 1e-9)
        << "residual changed under a constant shift — not translation-invariant";
}

// A4. Scaling the field scales the residual by the same factor.
// Linearity: || A(alpha*T) ||_inf = |alpha| * || A(T) ||_inf.
TEST(ResidualOperator, ScalesLinearly) {
    const int N = 24;
    const double alpha = 3.5;
    core::Grid2D a = fresh(N);
    core::Grid2D b(N, N);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            b(i, j) = alpha * a(i, j);
    const double ra = core::laplace_residual_linf(a);
    const double rb = core::laplace_residual_linf(b);
    EXPECT_NEAR(rb, alpha * ra, 1e-9 + 1e-9 * std::abs(alpha * ra))
        << "residual does not scale linearly with the field";
}

// ════════════════════════════════════════════════════════════════════════════
// PART B — convergence vs iteration (JacobiCPU, measured by the unified residual)
// ════════════════════════════════════════════════════════════════════════════

// B1. The equation residual must be monotonically non-increasing.
// Jacobi on the Laplace BVP is a contraction; the residual cannot grow
// (1% slack absorbs floating-point noise). A rise signals divergence.
TEST(Convergence, EquationResidualIsMonotoneNonIncreasing) {
    const auto h = jacobi_equation_history(/*N=*/24, /*steps=*/400);
    ASSERT_GT(h.size(), 1u);
    for (size_t k = 1; k < h.size(); ++k)
        EXPECT_LE(h[k], h[k - 1] * 1.01)
            << "equation residual increased at iteration " << k
            << " (" << h[k - 1] << " -> " << h[k] << ")";
}

// B2. A tighter tolerance must never converge in fewer iterations.
// From one history: iters(tol) = first k with r_k/r_0 < tol. Monotone iters
// vs decreasing tol is a basic correctness property of a convergent iteration.
TEST(Convergence, TighterToleranceNeedsMoreIterations) {
    const auto h = jacobi_equation_history(/*N=*/24, /*steps=*/6000);
    ASSERT_FALSE(h.empty());
    const double r0 = h.front();

    auto iters_for = [&](double tol) -> int {
        for (size_t k = 0; k < h.size(); ++k)
            if (h[k] / r0 < tol) return static_cast<int>(k);
        return static_cast<int>(h.size());   // not reached within the run
    };

    const int k_3 = iters_for(1e-3);
    const int k_5 = iters_for(1e-5);
    const int k_7 = iters_for(1e-7);

    EXPECT_LE(k_3, k_5) << "1e-5 converged in fewer iters than 1e-3";
    EXPECT_LE(k_5, k_7) << "1e-7 converged in fewer iters than 1e-5";
    EXPECT_GT(k_7, k_3) << "tolerance had no effect on iteration count";
}

// B3. THE CFD-grade test: the asymptotic per-iteration residual reduction must
// match Jacobi's known spectral radius for this grid,
//     rho = cos(pi / (N - 1)).
// After the fast error modes decay, r_{k+1}/r_k -> rho. We use a SMALL grid so
// rho sits well below 1 and is measurable, take the geometric mean of the
// ratio over a post-transient, pre-floor window, and require it near theory.
TEST(Convergence, AsymptoticRateMatchesJacobiSpectralRadius) {
    const int N = 8;                                  // rho = cos(pi/7) ~ 0.9010
    const double rho_theory = std::cos(PI / (N - 1));

    const auto h = jacobi_equation_history(N, /*steps=*/70);
    ASSERT_GE(h.size(), 61u);

    // Geometric mean of r_{k+1}/r_k over [lo, hi): smooths L-inf wobble and the
    // last sub-dominant mode (≈1% by k=25), while staying above the FP floor.
    const int lo = 25, hi = 60;
    double log_sum = 0.0; int count = 0;
    for (int k = lo; k < hi; ++k) {
        if (h[k] > 0.0 && h[k + 1] > 0.0) {
            log_sum += std::log(h[k + 1] / h[k]);
            ++count;
        }
    }
    ASSERT_GT(count, 0);
    const double rho_measured = std::exp(log_sum / count);

    EXPECT_NEAR(rho_measured, rho_theory, 0.06)
        << "measured Jacobi convergence factor " << rho_measured
        << " differs from theory cos(pi/(N-1)) = " << rho_theory
        << " — wrong stencil, BC handling, or not actually Jacobi";
    EXPECT_LT(rho_measured, 1.0)
        << "iteration is not contracting (rho >= 1) — divergence";
}

// B4. The pointwise identity equation = 4 x increment must hold at EVERY
// iteration, not just the first. (Jacobi update = 1/4 of the neighbour sum.)
TEST(Convergence, EquationEqualsFourTimesIncrementEveryStep) {
    const int N = 24;
    core::Grid2D g = fresh(N);
    solver::JacobiCPU s;
    for (int k = 0; k < 300; ++k) {
        const double eq_before = core::laplace_residual_linf(g);   // residual of T^k
        s.step(g);
        const double increment = s.residual();                     // ||T^{k+1}-T^k||
        EXPECT_NEAR(eq_before, 4.0 * increment,
                    1e-9 + 1e-9 * std::abs(eq_before))
            << "identity broke at iteration " << k
            << ": eq=" << eq_before << " 4*incr=" << 4.0 * increment;
    }
}
