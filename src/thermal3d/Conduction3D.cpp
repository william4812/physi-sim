#include "thermal3d/Conduction3D.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace physi_sim::thermal3d {

Conduction3D::Conduction3D(std::size_t n, double length, double k, double h)
    : n_(n), L_(length), k_(k), h_(h), dx_(length / static_cast<double>(n)),
      T_(n * n * n, 0.0), aP_(n * n * n, 0.0), rhs_(n * n * n, 0.0),
      q_([](double, double, double) { return 0.0; }),
      tinf_([](double, double, double) { return 0.0; }) {
    if (n < 2) throw std::invalid_argument("Conduction3D: need n >= 2");
    if (k <= 0.0 || h <= 0.0 || length <= 0.0)
        throw std::invalid_argument("Conduction3D: k, h, length must be > 0");
}

void Conduction3D::setSource(ScalarField q)     { q_ = std::move(q);    dirty_ = true; }
void Conduction3D::setAmbient(ScalarField tinf) { tinf_ = std::move(tinf); dirty_ = true; }
void Conduction3D::setInitial(double t0)        { std::fill(T_.begin(), T_.end(), t0); }

void Conduction3D::assemble() {
    const double kf  = k_ * dx_;                                    // interior face conductance
    const double ab  = 2.0 * k_ * h_ * dx_ * dx_ / (2.0 * k_ + h_ * dx_);  // Robin face conductance
    const double vol = dx_ * dx_ * dx_;

    for (std::size_t i = 0; i < n_; ++i)
        for (std::size_t j = 0; j < n_; ++j)
            for (std::size_t k = 0; k < n_; ++k) {
                const double x = center(i), y = center(j), z = center(k);
                double aP  = 0.0;
                double rhs = q_(x, y, z) * vol;

                if (i > 0)      aP += kf; else { aP += ab; rhs += ab * tinf_(0.0, y, z); }
                if (i + 1 < n_) aP += kf; else { aP += ab; rhs += ab * tinf_(L_,  y, z); }
                if (j > 0)      aP += kf; else { aP += ab; rhs += ab * tinf_(x, 0.0, z); }
                if (j + 1 < n_) aP += kf; else { aP += ab; rhs += ab * tinf_(x, L_,  z); }
                if (k > 0)      aP += kf; else { aP += ab; rhs += ab * tinf_(x, y, 0.0); }
                if (k + 1 < n_) aP += kf; else { aP += ab; rhs += ab * tinf_(x, y, L_ ); }

                aP_[idx(i, j, k)]  = aP;
                rhs_[idx(i, j, k)] = rhs;
            }
    dirty_ = false;
}

std::size_t Conduction3D::solve(double tol, std::size_t maxIter) {
    //return 0; // Temporary - Red step
    if (dirty_) assemble();
    const double kf = k_ * dx_;
    const double pi = 3.14159265358979323846;
    const double omega = 2.0 / (1.0 + std::sin(pi / static_cast<double>(n_)));  // optimal-ish SOR

    for (std::size_t it = 1; it <= maxIter; ++it) {
        double maxDelta = 0.0;
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = 0; j < n_; ++j)
                for (std::size_t k = 0; k < n_; ++k) {
                    const std::size_t p = idx(i, j, k);
                    double nb = 0.0;
                    if (i > 0)      nb += kf * T_[idx(i - 1, j, k)];
                    if (i + 1 < n_) nb += kf * T_[idx(i + 1, j, k)];
                    if (j > 0)      nb += kf * T_[idx(i, j - 1, k)];
                    if (j + 1 < n_) nb += kf * T_[idx(i, j + 1, k)];
                    if (k > 0)      nb += kf * T_[idx(i, j, k - 1)];
                    if (k + 1 < n_) nb += kf * T_[idx(i, j, k + 1)];

                    const double tNew = (nb + rhs_[p]) / aP_[p];
                    const double d    = omega * (tNew - T_[p]);
                    T_[p] += d;
                    maxDelta = std::max(maxDelta, std::abs(d));
                }
        if (maxDelta < tol) return it;
    }
    return maxIter;
}

double Conduction3D::energyImbalance() const {
    const double ab  = 2.0 * k_ * h_ * dx_ * dx_ / (2.0 * k_ + h_ * dx_);
    const double vol = dx_ * dx_ * dx_;
    double net = 0.0;
    for (std::size_t i = 0; i < n_; ++i)
        for (std::size_t j = 0; j < n_; ++j)
            for (std::size_t k = 0; k < n_; ++k) {
                const double x = center(i), y = center(j), z = center(k);
                const double Tp = T_[idx(i, j, k)];
                net += q_(x, y, z) * vol;                                    // generated
                if (i == 0)      net += ab * (tinf_(0.0, y, z) - Tp);        // + boundary influx
                if (i + 1 == n_) net += ab * (tinf_(L_,  y, z) - Tp);
                if (j == 0)      net += ab * (tinf_(x, 0.0, z) - Tp);
                if (j + 1 == n_) net += ab * (tinf_(x, L_,  z) - Tp);
                if (k == 0)      net += ab * (tinf_(x, y, 0.0) - Tp);
                if (k + 1 == n_) net += ab * (tinf_(x, y, L_ ) - Tp);
            }
    return net;
}

}  // namespace physi_sim::thermal3d
