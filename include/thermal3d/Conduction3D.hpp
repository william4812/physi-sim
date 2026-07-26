#pragma once
#include <cstddef>
#include <functional>
#include <vector>

namespace physi_sim::thermal3d {

/// Steady 3D heat conduction on a uniform, cell-centered finite-volume grid.
///
///     -div(k grad T) = q            in (0,L)^3
///     -k dT/dn = h (T_s - T_inf)    on all six faces   (Robin / convective)
///
/// Constant k. Second-order accurate in dx (verified by the Method of
/// Manufactured Solutions). The Robin face uses the exact series resistance
/// of the half-cell conduction and the film coefficient:
///     U = 1 / (dx/(2k) + 1/h) = 2kh / (2k + h*dx)
class Conduction3D {
public:
    using ScalarField = std::function<double(double, double, double)>;

    Conduction3D(std::size_t n, double length, double k, double h);

    void setSource(ScalarField q);      ///< q [W/m^3], sampled at cell centers
    void setAmbient(ScalarField tinf);  ///< T_inf [K], sampled at boundary face centers
    void setInitial(double t0);

    /// Iterate (SOR) to the steady discrete solution. Returns iterations used.
    std::size_t solve(double tol = 1e-9, std::size_t maxIter = 200000);

    double at(std::size_t i, std::size_t j, std::size_t k) const { return T_[idx(i, j, k)]; }
    double center(std::size_t i) const { return (static_cast<double>(i) + 0.5) * dx_; }
    std::size_t n() const { return n_; }
    double dx() const { return dx_; }

    /// Net power imbalance [W]: (total source) + (net boundary influx).
    /// Interior fluxes cancel pairwise, so this is ~0 at steady state.
    double energyImbalance() const;

private:
    std::size_t idx(std::size_t i, std::size_t j, std::size_t k) const { return (i * n_ + j) * n_ + k; }
    void assemble();

    std::size_t n_;
    double L_, k_, h_, dx_;
    std::vector<double> T_, aP_, rhs_;
    ScalarField q_, tinf_;
    bool dirty_ = true;
};

}  // namespace physi_sim::thermal3d
