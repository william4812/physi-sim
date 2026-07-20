// tests/verification/test_thermal_conduction_physics.cpp
//
// BOTTOM-UP ANALYTICAL BENCHMARK LADDER for solveField().
//
// Every level adds EXACTLY ONE new term to the previous one, and every level has
// a closed-form reference. If a level fails, the term it introduced is the
// suspect -- nothing else changed.
//
//   L0  plumbing            uniform Dirichlet, no source        EXACT
//   L1  + conduction        1D linear profile                   EXACT
//   L2  + material jump     Si|TIM harmonic mean                EXACT
//   L3  + convective BC     Dirichlet | Robin                   EXACT
//   L4  + volumetric source 1D parabola                         EXACT vs DISCRETE
//   L5  + 3D spreading      3D parabolic source                 2nd ORDER
//   L6  global invariant    energy balance                      EXACT
//
// ---------------------------------------------------------------------------
// WHICH LEVELS ARE EXACT, AND WHY -- read this before changing a tolerance.
//
// A cell-centred finite-volume unknown is the CELL AVERAGE of the field over its
// control volume, not the point value at the centre. For a LINEAR field the two
// coincide, and the two-point face gradient is also exact, so levels L0-L3 are
// reproduced to ROUND-OFF: assert 1e-9, not "close enough".
//
// A volumetric source curves the profile, and two separate O(dx^2) gaps open up:
//   (a) cell average != centre value      (differ by T'' dx^2/24)
//   (b) the half-cell Dirichlet conductance assumes a LINEAR profile over the
//       last half cell, which a parabola is not.
// So a source term makes the scheme SECOND-ORDER CONVERGENT, not exact, when
// compared against the analytic CONTINUUM solution.
//
// In 1D that error has a closed form. Substituting T_i = a z_i (L - z_i) + c
// with a = qv/2k satisfies every INTERIOR equation for any c; the boundary cell
// equation -3 T_0 + T_1 + qv dx^2 / k = 0 then fixes c = qv dx^2 / (8k). So the
// EXACT DISCRETE solution is known in closed form and L4 can assert machine
// precision after all. In 3D no such closed form exists, so L5 must assert the
// CONVERGENCE ORDER instead.
//
// ---------------------------------------------------------------------------
// DIMENSIONAL AUDIT of solveField(gamma, bc, src, phi, ...)
//
//   symbol         quantity                    unit (thermal)     unit (electrical)
//   -------------  --------------------------  -----------------  ------------------
//   gamma[c]       conductivity                W/(m K)            S/m
//   src[c]         VOLUMETRIC source           W/m^3              A/m^3
//   phi[c]         potential                   K                  V
//   dx             cell size                   m                  m
//   faceArea=dx^2  face area                   m^2                m^2
//   cellVolume=dx^3                            m^3                m^3
//
//   interior face conductance   g = gamma * dx      [W/(m K)][m] = W/K          [S]
//   Dirichlet face conductance  g = 2 * gamma * dx  same                        [S]
//   Robin face conductance      U*A = A / (dx/(2 gamma) + 1/h)
//                               [m^2] / ([m][m K/W] + [m^2 K/W]) = W/K          --
//   rhs from source             src * dx^3          [W/m^3][m^3] = W            [A]
//   rhs from Neumann            value * dx^2        [W/m^2][m^2] = W            [A]
//   discrete equation           diag * phi = rhs    [W/K][K] = W                [S][V]=[A]
//
// Every term in the assembled equation is a POWER in watts (thermal) or a
// CURRENT in amperes (electrical). That is the balance being enforced per cell,
// and it is why `src` must be per unit VOLUME, never a total.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "thermal3d/ElectroThermal3D.hpp"

using namespace physi_sim::thermal3d;

namespace {

using BCs = ElectroThermal3D::BCs;
using FB  = ElectroThermal3D::FaceBC;

/// All six faces adiabatic (zero inward flux). The starting point for every 1D
/// case: insulating the four side walls is what reduces a 3D solver to a 1D
/// problem, and it is also the unit-cell symmetry condition of a periodic array.
BCs allAdiabatic() {
    BCs b;
    for (auto& f : b) {
        f.type  = FB::Neumann;
        f.value = [](double, double, double) { return 0.0; };
    }
    return b;
}

void setDirichlet(BCs& b, std::size_t face, double value) {
    b[face].type  = FB::Dirichlet;
    b[face].value = [value](double, double, double) { return value; };
}

std::size_t flat(std::size_t n, std::size_t i, std::size_t j, std::size_t k) {
    return (i * n + j) * n + k;   // must match ElectroThermal3D::idx
}

/// Solve  div(k grad T) + src = 0  and return the field. `src` is VOLUMETRIC [W/m^3].
std::vector<double> solveThermalField(std::size_t n, double L,
                                      ElectroThermal3D::ScalarField k,
                                      const BCs& bcs,
                                      const std::vector<double>& src,
                                      double tol = 1e-12) {
    // The sigma argument is unused by solveThermal but must be strictly positive.
    ElectroThermal3D solver(n, L, [](double, double, double) { return 1.0; });
    std::vector<double> T(n * n * n, 0.0);
    EXPECT_TRUE(solver.solveThermal(k, bcs, src, T, tol))
        << "thermal solve failed to converge at n=" << n;
    return T;
}

constexpr double EXACT_TOL = 1e-9;   // round-off band for linear-field levels

}  // namespace

// ===========================================================================
// L0 -- PLUMBING. No source, every face held at the same temperature.
// The only field satisfying this is the constant. Verifies indexing, boundary
// dispatch and the Dirichlet coefficient without exercising any gradient.
// ===========================================================================
TEST(ThermalLadderL0, UniformDirichletGivesUniformField) {
    const std::size_t n = 8;
    const double L = 1.0, T_wall = 300.0;      // dummy values: k = 1, L = 1

    BCs bcs;
    for (std::size_t f = 0; f < ElectroThermal3D::FaceCount; ++f) setDirichlet(bcs, f, T_wall);

    const auto T = solveThermalField(n, L, [](double, double, double) { return 1.0; },
                                     bcs, std::vector<double>(n * n * n, 0.0));

    double worst = 0.0;
    for (double t : T) worst = std::max(worst, std::abs(t - T_wall));
    EXPECT_LT(worst, EXACT_TOL) << "Linf = " << worst << " K from a constant field.";
}

// ===========================================================================
// L1 -- CONDUCTION. Adds a gradient. Dirichlet on the two z-faces, adiabatic
// sides, no source: Fourier's law with constant flux gives a straight line.
//
//     T(z) = T_hot - (T_hot - T_cold) z / L
//     q''  = k (T_hot - T_cold) / L                       [W/m^2]
//
// A LINEAR field, so the FV scheme is EXACT: cell average equals centre value
// and the two-point face gradient is the true gradient.
// ===========================================================================
TEST(ThermalLadderL1, LinearProfileIsExact) {
    const double L = 1.0, k = 1.0, T_hot = 400.0, T_cold = 300.0;

    for (std::size_t n : {8u, 16u}) {
        BCs bcs = allAdiabatic();
        setDirichlet(bcs, ElectroThermal3D::ZLow,  T_hot);
        setDirichlet(bcs, ElectroThermal3D::ZHigh, T_cold);

        const auto T = solveThermalField(n, L, [k](double, double, double) { return k; },
                                         bcs, std::vector<double>(n * n * n, 0.0));
        const double dx = L / static_cast<double>(n);

        double worst = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t kk = 0; kk < n; ++kk) {
                    const double z = (static_cast<double>(kk) + 0.5) * dx;
                    const double exact = T_hot - (T_hot - T_cold) * z / L;
                    worst = std::max(worst, std::abs(T[flat(n, i, j, kk)] - exact));
                }
        EXPECT_LT(worst, EXACT_TOL)
            << "n=" << n << ": Linf = " << worst << " K. A linear field must be exact; "
               "any error here is in the Dirichlet coefficient or the face gradient.";
    }
}

// ===========================================================================
// L2 -- MATERIAL INTERFACE. Adds a discontinuity in k. Real properties now:
// a thinned silicon die bonded through a thermal interface material.
//
// Steady, no source => the flux q'' is IDENTICAL in both layers, so
//     dT/dz = -q''/k
// and the same flux produces a slope 30x steeper in the TIM. Temperature stays
// continuous; its GRADIENT jumps by the conductivity ratio.
//
//     R'' = t_Si/k_Si + t_TIM/k_TIM      q'' = (T_hot - T_cold) / R''
//
// EXACT: at a face-aligned interface the harmonic mean is not an approximation,
// it IS the series resistance. This also pins the physics that makes accelerators
// throttle -- the TIM carries ~91% of the junction-to-case temperature drop.
// ===========================================================================
TEST(ThermalLadderL2, SiliconTimSeriesResistanceIsExact) {
    const double t_Si = 300e-6, t_TIM = 100e-6;      // m
    const double L = t_Si + t_TIM;
    const double k_Si = 150.0, k_TIM = 5.0;          // W/(m K)
    const double T_hot = 383.15, T_cold = 317.15;    // K  (110 C junction, 44 C case)

    const double R_total = t_Si / k_Si + t_TIM / k_TIM;          // m^2 K/W
    const double flux    = (T_hot - T_cold) / R_total;           // W/m^2
    const double dT_Si   = flux * t_Si / k_Si;                   // K
    const double dT_TIM  = flux * t_TIM / k_TIM;                 // K

    // Sanity on the analytic reference itself before trusting it as an oracle.
    ASSERT_NEAR(dT_Si + dT_TIM, T_hot - T_cold, 1e-9);
    EXPECT_GT(dT_TIM / (dT_Si + dT_TIM), 0.85)
        << "the TIM must dominate junction-to-case resistance";

    for (std::size_t n : {8u, 16u}) {
        BCs bcs = allAdiabatic();
        setDirichlet(bcs, ElectroThermal3D::ZLow,  T_hot);
        setDirichlet(bcs, ElectroThermal3D::ZHigh, T_cold);

        auto kField = [t_Si, k_Si, k_TIM](double, double, double z) {
            return z < t_Si ? k_Si : k_TIM;
        };
        const auto T = solveThermalField(n, L, kField, bcs, std::vector<double>(n * n * n, 0.0));
        const double dx = L / static_cast<double>(n);

        double worst = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t kk = 0; kk < n; ++kk) {
                    const double z = (static_cast<double>(kk) + 0.5) * dx;
                    const double exact = (z < t_Si)
                        ? T_hot - flux * z / k_Si
                        : T_hot - dT_Si - flux * (z - t_Si) / k_TIM;
                    worst = std::max(worst, std::abs(T[flat(n, i, j, kk)] - exact));
                }
        EXPECT_LT(worst, 1e-8)
            << "n=" << n << ": Linf = " << worst << " K. An arithmetic face mean would give "
               "k_face = 77.5 instead of 9.68 at the Si|TIM face -- 8x too conductive.";
    }
}

// ===========================================================================
// L3 -- CONVECTIVE BOUNDARY. Adds a film resistance in series with conduction.
// A boundary is just a face whose neighbour is a fluid: 1/h replaces dx/(2k).
//
//     R'' = L/k + 1/h        q'' = (T_hot - T_inf) / R''
//     T(z) = T_hot - q'' z / k        film drop at the wall = q''/h
//
// Still a LINEAR field, so still EXACT.
// ===========================================================================
TEST(ThermalLadderL3, DirichletToRobinIsExact) {
    const double L = 1e-3, k = 150.0, h = 1e5;       // W/(m^2 K) direct-to-chip coldplate
    const double T_hot = 400.0, T_inf = 300.0;

    const double R_total = L / k + 1.0 / h;
    const double flux    = (T_hot - T_inf) / R_total;

    for (std::size_t n : {8u, 16u}) {
        BCs bcs = allAdiabatic();
        setDirichlet(bcs, ElectroThermal3D::ZLow, T_hot);
        bcs[ElectroThermal3D::ZHigh].type  = FB::Robin;
        bcs[ElectroThermal3D::ZHigh].h     = h;
        bcs[ElectroThermal3D::ZHigh].value = [T_inf](double, double, double) { return T_inf; };

        const auto T = solveThermalField(n, L, [k](double, double, double) { return k; },
                                         bcs, std::vector<double>(n * n * n, 0.0));
        const double dx = L / static_cast<double>(n);

        double worst = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t kk = 0; kk < n; ++kk) {
                    const double z = (static_cast<double>(kk) + 0.5) * dx;
                    worst = std::max(worst, std::abs(T[flat(n, i, j, kk)] - (T_hot - flux * z / k)));
                }
        EXPECT_LT(worst, 1e-8) << "n=" << n << ": Linf = " << worst << " K.";
    }
}

// ===========================================================================
// L4 -- VOLUMETRIC SOURCE. Adds q_v. The profile becomes a parabola:
//
//     T(z) = q_v z (L - z) / (2k)          T_max = q_v L^2 / (8k)
//
// This is the level that most often produces a confusing failure, because the
// discrete solution does NOT equal the analytic point values -- it is offset by
// a known constant. Substituting T_i = a z_i (L - z_i) + c with a = q_v/2k
// satisfies every interior equation for any c; the first-cell equation
//
//     -3 T_0 + T_1 + q_v dx^2 / k = 0
//
// pins c = q_v dx^2 / (8k). Comparing against the EXACT DISCRETE solution makes
// this level machine-exact; comparing against the continuum solution would only
// be second-order and would fail any tight tolerance.
// ===========================================================================
TEST(ThermalLadderL4, UniformSourceMatchesExactDiscreteSolution) {
    const double L = 1e-3, k = 1.0;
    const double Q_total = 1e-2;                       // W deposited in the whole cube
    const double q_v = Q_total / (L * L * L);          // W/m^3  -- VOLUMETRIC, per the API

    ASSERT_DOUBLE_EQ(q_v, 1e7);                        // dimensional sanity on the reference

    for (std::size_t n : {8u, 16u, 32u}) {
        BCs bcs = allAdiabatic();                      // sides insulated -> genuinely 1D
        setDirichlet(bcs, ElectroThermal3D::ZLow,  0.0);
        setDirichlet(bcs, ElectroThermal3D::ZHigh, 0.0);

        const auto T = solveThermalField(n, L, [k](double, double, double) { return k; },
                                         bcs, std::vector<double>(n * n * n, q_v));
        const double dx = L / static_cast<double>(n);
        const double c  = q_v * dx * dx / (8.0 * k);   // exact discrete offset

        double worst = 0.0;

        // Since sides are insulated and 1D, we can pick any (i, j) column (e.g., i = 0, j = 0)
        // to inspect profiles cleanly along the z-direction (kk).
        const std::size_t i = 0;
        const std::size_t j = 0;

        std::printf("\n--- Grid Resolution n = %zu (dx = %.5e m, c = %.6e K) ---\n", n, dx, c);
        std::printf("  %6s %16s %16s %16s %16s\n", "z_idx", "z [mm]", "Numerical [K]", "Exact Disc [K]", "Error [K]");

        for (std::size_t kk = 0; kk < n; ++kk) {
            const double z = (static_cast<double>(kk) + 0.5) * dx;
            const double exactDiscrete = q_v * z * (L - z) / (2.0 * k) + c;
            const double numerical = T[flat(n, i, j, kk)];
            const double err = std::abs(numerical - exactDiscrete);

            worst = std::max(worst, err);

            std::printf("  %6zu %16.6f %16.9f %16.9f %16.3e\n",
                        kk, z * 1e3, numerical, exactDiscrete, err);
        }

        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t kk = 0; kk < n; ++kk) {
                    const double z = (static_cast<double>(kk) + 0.5) * dx;
                    const double exactDiscrete = q_v * z * (L - z) / (2.0 * k) + c;
                    worst = std::max(worst, std::abs(T[flat(n, i, j, kk)] - exactDiscrete));
                }
        EXPECT_LT(worst, 1e-9)
            << "n=" << n << ": Linf = " << worst << " K against the exact discrete solution "
               "(offset c = " << c << " K).";
    }
}

// ===========================================================================
// L5 -- THREE-DIMENSIONAL SPREADING. Same source as L4, but now ALL SIX faces
// are held at zero, so heat escapes in three directions instead of one. The
// centre temperature is therefore MUCH lower than the 1D slab formula predicts
// -- about 45% of it. Using the 1D result here is the single most common error
// in this ladder.
//
// The exact solution is a triple Fourier series with no elementary closed form,
// so this level asserts SECOND-ORDER CONVERGENCE rather than exactness.
//
//     T = (q_v/k) L^2 u(x/L, y/L, z/L),   laplacian(u) = -1, u = 0 on the boundary
//     u = sum over odd m,n,p of 64 / (pi^5 m n p (m^2+n^2+p^2)) sin sin sin
//
// Reference values below are that series summed to 401 terms per index at each
// grid's probe point (drift below 3e-8, far under the discretisation error).
// ===========================================================================
TEST(ThermalLadderL5, ThreeDimensionalSourceConvergesAtSecondOrder) {
    const double L = 1e-3, k = 1.0, q_v = 1e7;

    struct Ref { std::size_t n; double exact; };
    const std::vector<Ref> refs{{8u, 0.542807733}, {16u, 0.557258697}, {32u, 0.560908400}};

    std::vector<double> err;
    std::printf("\n  L5  3D uniform source, all six faces Dirichlet 0 K\n"
                "      %5s %16s %16s %14s %8s\n", "n", "T_centre [K]", "exact 3D [K]", "rel err", "order");

    for (const auto& r : refs) {
        const std::size_t n = r.n;
        BCs bcs;
        for (std::size_t f = 0; f < ElectroThermal3D::FaceCount; ++f) setDirichlet(bcs, f, 0.0);

        const auto T = solveThermalField(n, L, [k](double, double, double) { return k; },
                                         bcs, std::vector<double>(n * n * n, q_v), 1e-11);
        const std::size_t m = n / 2;
        const double got = T[flat(n, m, m, m)];
        const double e   = std::abs(got - r.exact) / r.exact;
        err.push_back(e);

        const double order = (err.size() < 2) ? 0.0 : std::log2(err[err.size() - 2] / e);
        if (err.size() < 2) std::printf("      %5zu %16.9f %16.9f %14.3e %8s\n", n, got, r.exact, e, "-");
        else                std::printf("      %5zu %16.9f %16.9f %14.3e %8.3f\n", n, got, r.exact, e, order);
    }
    std::printf("\n");

    for (std::size_t i = 1; i < err.size(); ++i) {
        const double order = std::log2(err[i - 1] / err[i]);
        EXPECT_GT(order, 1.8) << "observed order " << order << ", expected ~2";
        EXPECT_LT(order, 2.3) << "observed order " << order << ", expected ~2";
    }
}

// ===========================================================================
// L6 -- GLOBAL INVARIANT. The First Law over the whole domain, independent of
// mesh: everything generated inside must leave through the boundary.
//
//     q_v * L^3  ==  sum over boundary faces of  G (T_P - T_wall)
//
// A finite-volume scheme sums fluxes over faces, so interior fluxes cancel
// pairwise BY CONSTRUCTION. Any imbalance is a defect, not an approximation --
// hence a round-off tolerance regardless of resolution.
// ===========================================================================
TEST(ThermalLadderL6, GlobalEnergyBalanceClosesToRoundOff) {
    const std::size_t n = 16;
    const double L = 1e-3, k = 1.0, q_v = 1e7;

    BCs bcs = allAdiabatic();                    // only the two z-faces can carry heat out
    setDirichlet(bcs, ElectroThermal3D::ZLow,  0.0);
    setDirichlet(bcs, ElectroThermal3D::ZHigh, 0.0);

    const auto T = solveThermalField(n, L, [k](double, double, double) { return k; },
                                     bcs, std::vector<double>(n * n * n, q_v));
    const double dx = L / static_cast<double>(n);

    const double generated = q_v * L * L * L;    // [W/m^3][m^3] = W
    const double gBoundary = 2.0 * k * dx;       // half-cell conductance [W/K]
    double efflux = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            efflux += gBoundary * (T[flat(n, i, j, 0)]     - 0.0);
            efflux += gBoundary * (T[flat(n, i, j, n - 1)] - 0.0);
        }

    EXPECT_NEAR(efflux, generated, 1e-9 * generated)
        << "generated " << generated << " W but " << efflux << " W left the domain.";
}
