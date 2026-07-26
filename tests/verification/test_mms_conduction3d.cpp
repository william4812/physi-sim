// tests/verification/test_mms_conduction3d.cpp
//
// VERIFICATION (the "V" in V&V): does the code solve the equations correctly,
// at its formal order of accuracy? This is a different question from unit tests
// (does the class work), integration (do components compose), and regression
// (did results change) -- hence its own test category.
//
// Method of Manufactured Solutions (MMS):
//   1. Choose a smooth field T*(x,y,z).
//   2. Substitute it into the PDE to get the exact source q* that makes T* true.
//   3. Impose the exactly consistent Robin ambient T_inf on every boundary face.
//   4. Require the solver to recover T* AT ITS FORMAL ORDER (2nd: halve dx -> err/4).
//
// The ORDER, not the error value, is what catches bugs: a single-grid check can't
// distinguish a correct scheme from a subtly broken one, but one wrong coefficient
// collapses the observed order to 1 (or 0).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "thermal3d/Conduction3D.hpp"

namespace {

// ---- Manufactured solution -------------------------------------------------
// T*(x,y,z) = T0 + A sin(Wx+p) sin(Wy+p) sin(Wz+p)
// The phase p is nonzero on purpose: it keeps both T* and its gradient nonzero on
// every boundary, so the Robin condition is genuinely exercised (a solution that
// vanished on the walls would only test it trivially).
constexpr double L  = 1.0;      // m       domain edge
constexpr double K  = 150.0;    // W/m-K   silicon-like conductivity
constexpr double H  = 1000.0;   // W/m^2-K coldplate-like film coefficient
constexpr double A  = 50.0;     // K       amplitude
constexpr double T0 = 300.0;    // K       offset
constexpr double PH = 0.3;      // rad     phase
const double W = 1.7 * 3.14159265358979323846;

double Tex (double x, double y, double z) { return T0 + A * std::sin(W*x+PH) * std::sin(W*y+PH) * std::sin(W*z+PH); }
double dTdx(double x, double y, double z) { return A*W * std::cos(W*x+PH) * std::sin(W*y+PH) * std::sin(W*z+PH); }
double dTdy(double x, double y, double z) { return A*W * std::sin(W*x+PH) * std::cos(W*y+PH) * std::sin(W*z+PH); }
double dTdz(double x, double y, double z) { return A*W * std::sin(W*x+PH) * std::sin(W*y+PH) * std::cos(W*z+PH); }

// lap(T*) = -3W^2 (T* - T0)   =>   q* = -k lap(T*) = 3 k W^2 (T* - T0)
double Qsrc(double x, double y, double z) {
    return 3.0 * K * W * W * A * std::sin(W*x+PH) * std::sin(W*y+PH) * std::sin(W*z+PH);
}

// Robin BC:  -k dT/dn = h (T_s - T_inf)   =>   T_inf = T_s + (k/h) dT/dn
// dT/dn is the OUTWARD normal derivative on whichever face the point lies.
double Tinf(double x, double y, double z) {
    const double e = 1e-12;
    double dn = 0.0;
    if      (std::abs(x)     < e) dn = -dTdx(x, y, z);   // x-low  face, n = -x
    else if (std::abs(x - L) < e) dn = +dTdx(x, y, z);   // x-high face, n = +x
    else if (std::abs(y)     < e) dn = -dTdy(x, y, z);
    else if (std::abs(y - L) < e) dn = +dTdy(x, y, z);
    else if (std::abs(z)     < e) dn = -dTdz(x, y, z);
    else if (std::abs(z - L) < e) dn = +dTdz(x, y, z);
    return Tex(x, y, z) + (K / H) * dn;
}

// Volume-weighted L2 norm of (numerical - exact) over the domain.
double l2Error(std::size_t n, std::size_t* iters = nullptr) {
    physi_sim::thermal3d::Conduction3D s(n, L, K, H);
    s.setSource(Qsrc);
    s.setAmbient(Tinf);
    s.setInitial(T0);
    const std::size_t it = s.solve(1e-9, 500000);
    if (iters) *iters = it;

    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < n; ++k) {
                const double err = s.at(i, j, k) - Tex(s.center(i), s.center(j), s.center(k));
                sum += err * err;
            }
    const double dx = s.dx();
    return std::sqrt(sum * dx * dx * dx);
}

}  // namespace

// ---------------------------------------------------------------------------
// The order-of-accuracy test -- the core credential.
// Expected (measured): order 2.04 (8->16), 2.01 (16->32). Asserted band [1.8, 2.2]
// is tight enough to catch a first-order bug (almost always the boundary
// treatment) and loose enough to allow the asymptotic approach onto 2.
// ---------------------------------------------------------------------------
TEST(MMSConduction3D, SecondOrderSpatialConvergence) {
    const std::vector<std::size_t> grids{8, 16, 32};
    std::vector<double> err;
    std::vector<std::size_t> its;

    for (std::size_t n : grids) {
        std::size_t it = 0;
        err.push_back(l2Error(n, &it));
        its.push_back(it);
    }

    std::cout << "\n  MMS convergence -- 3D conduction + Robin BC\n"
              << "     N        dx        L2 error     order    SOR iters\n";
    for (std::size_t i = 0; i < grids.size(); ++i) {
        const std::string ord =
            (i == 0) ? "-" : std::to_string(std::log2(err[i - 1] / err[i])).substr(0, 5);
        std::printf("  %4zu  %8.5f  %12.4e  %7s  %9zu\n",
                    grids[i], L / static_cast<double>(grids[i]), err[i], ord.c_str(), its[i]);
    }
    std::cout << std::endl;

    for (std::size_t i = 1; i < err.size(); ++i) {
        EXPECT_LT(err[i], err[i - 1]) << "error must decrease under grid refinement";
        const double order = std::log2(err[i - 1] / err[i]);
        EXPECT_GT(order, 1.8) << "observed order " << order << " (expected ~2), N="
                              << grids[i - 1] << "->" << grids[i];
        EXPECT_LT(order, 2.2) << "observed order " << order << " (expected ~2), N="
                              << grids[i - 1] << "->" << grids[i];
    }
}

// ---------------------------------------------------------------------------
// Conservation -- a finite-volume scheme sums fluxes over faces, so interior
// fluxes cancel pairwise BY CONSTRUCTION. At steady state:
//     total generated power + net boundary influx = 0
// Any imbalance is a bug, full stop. (Measured: ~1e-10 relative.)
// ---------------------------------------------------------------------------
TEST(MMSConduction3D, ConservesEnergyToRoundOff) {
    const std::size_t n = 16;
    physi_sim::thermal3d::Conduction3D s(n, L, K, H);
    s.setSource(Qsrc);
    s.setAmbient(Tinf);
    s.setInitial(T0);
    s.solve(1e-10, 500000);

    const double dx = s.dx(), vol = dx * dx * dx;
    double scale = 0.0;  // total |generated power|, for a relative tolerance
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < n; ++k)
                scale += std::abs(Qsrc(s.center(i), s.center(j), s.center(k))) * vol;

    const double relImbalance = std::abs(s.energyImbalance()) / scale;
    EXPECT_LT(relImbalance, 1e-8) << "relative energy imbalance " << relImbalance;
}

// ---------------------------------------------------------------------------
// Invariance -- no source + uniform ambient must give a uniform field at T_inf.
// Cheap, and it catches whole classes of boundary-coefficient bugs.
// ---------------------------------------------------------------------------
TEST(MMSConduction3D, NoSourceUniformAmbientGivesUniformField) {
    const std::size_t n = 8;
    const double ambient = 350.0;
    physi_sim::thermal3d::Conduction3D s(n, L, K, H);
    s.setSource([](double, double, double) { return 0.0; });
    s.setAmbient([&](double, double, double) { return ambient; });
    s.setInitial(300.0);
    s.solve(1e-12, 500000);

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < n; ++k)
                EXPECT_NEAR(s.at(i, j, k), ambient, 1e-6);
}
