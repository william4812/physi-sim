/**
 * @file ElectroThermal3D.cpp
 * @brief Implementation of the 3D electro-thermal unit cell solver.
 */

#include "thermal3d/ElectroThermal3D.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace physi_sim::thermal3d {

ElectroThermal3D::ElectroThermal3D(std::size_t n, double length, ScalarField sigma)
    : n_(n), L_(length), dx_(length / static_cast<double>(n)), sigmaField_(std::move(sigma)) 
{
    if (n_ < 2) {
        throw std::invalid_argument("Grid resolution n must be >= 2.");
    }
    if (L_ <= 0.0) {
        throw std::invalid_argument("Domain length must be > 0.");
    }

    const std::size_t totalCells = n_ * n_ * n_;
    V_.resize(totalCells, 0.0);
    qv_.resize(totalCells, 0.0);

    // Default BCs: all Neumann (insulated)
    for (std::size_t f = 0; f < FaceCount; ++f) {
        ebc_[f].type = FaceBC::Neumann;
        ebc_[f].value = [](double, double, double) { return 0.0; };
        ebc_[f].h = 0.0;
    }
}

void ElectroThermal3D::setElectricalBC(const BCs& bc) {
    ebc_ = bc;
}

std::vector<double> ElectroThermal3D::sampleCells(const ScalarField& f) const {
    std::vector<double> data(n_ * n_ * n_);
    for (std::size_t i = 0; i < n_; ++i) {
        const double x = center(i);
        for (std::size_t j = 0; j < n_; ++j) {
            const double y = center(j);
            for (std::size_t k = 0; k < n_; ++k) {
                const double z = center(k);
                const double val = f(x, y, z);
                if (val <= 0.0) {
                    throw std::invalid_argument("Conductivity/Material field must be strictly positive (> 0).");
                }
                data[idx(i, j, k)] = val;
            }
        }
    }
    return data;
}

bool ElectroThermal3D::solve(double tol, std::size_t maxIter) {
    std::vector<double> sigma = sampleCells(sigmaField_);
    std::vector<double> zeroSrc(n_ * n_ * n_, 0.0); // No volumetric current source in DC via domain

    bool converged = solveField(sigma, ebc_, zeroSrc, V_, tol, maxIter);
    if (converged) {
        computeJoule(sigma);
    }
    return converged;
}

bool ElectroThermal3D::solveThermal(const ScalarField& k_field, const BCs& bcs,
                      const std::vector<double>& src, std::vector<double>& T,
                      double tol) 
{
        auto gamma = sampleCells(k_field);
        return solveField(gamma, bcs, src, T, tol, 500000);
    
}

bool ElectroThermal3D::solveField(const std::vector<double>& gamma, const BCs& bc,
                                  const std::vector<double>& src, std::vector<double>& phi,
                                  double tol, std::size_t maxIter) 
{
    const double dx2 = dx_ * dx_;
    const double faceArea = dx2; // dx^2
    const double cellVolume = dx2 * dx_; // dx^3
    const double omega = 1.6; // SOR over-relaxation parameter optimized for 3D meshes

    // Precalculate boundary flags and types for performance
    for (std::size_t iter = 0; iter < maxIter; ++iter) {
        double maxDelta = 0.0;

        for (std::size_t i = 0; i < n_; ++i) {
            const double x = center(i);
            for (std::size_t j = 0; j < n_; ++j) {
                const double y = center(j);
                for (std::size_t k = 0; k < n_; ++k) {
                    const double z = center(k); 
                    
                    const std::size_t c = idx(i, j, k);
                    const double g_c = gamma[c];

                    double diag = 0.0;
                    double rhs = src[c] * cellVolume;

                    // --- X-Axis Neighbors ---
                    if (i > 0) {
                        const std::size_t m = idx(i - 1, j, k);
                        const double g_face = dx_ * (2.0 * g_c * gamma[m]) / (g_c + gamma[m]); // Harmonic mean [W/K or S*m]
                        diag += g_face; // total conductance connected to your cell.
                        rhs += g_face * phi[m]; // the total current "weighted" by the neighbor potentials.
                    } else {
                        // XLow Boundary
                        const auto& fbc = bc[XLow];
                        if (fbc.type == FaceBC::Dirichlet) {
                            const double g_bound = 2.0 * g_c * dx_; // Half-cell conductance: k*A / (dx/2)
                            diag += g_bound;
                            rhs += g_bound * fbc.value(0.0, y, z);
                        } else if (fbc.type == FaceBC::Neumann) {
                            rhs += fbc.value(0.0, y, z) * faceArea; // Inward flux adds to RHS
                        } else if (fbc.type == FaceBC::Robin) {
                            const double u_bound = faceArea / ((dx_ / (2.0 * g_c)) + (1.0 / fbc.h));
                            diag += u_bound;
                            rhs += u_bound * fbc.value(0.0, y, z);
                        }
                    }

                    if (i + 1 < n_) {
                        const std::size_t m = idx(i + 1, j, k);
                        const double g_face = dx_ * (2.0 * g_c * gamma[m]) / (g_c + gamma[m]);
                        diag += g_face;
                        rhs += g_face * phi[m];
                    } else {
                        // XHigh Boundary
                        const auto& fbc = bc[XHigh];
                        if (fbc.type == FaceBC::Dirichlet) {
                            const double g_bound = 2.0 * g_c * dx_;
                            diag += g_bound;
                            rhs += g_bound * fbc.value(L_, y, z);
                        } else if (fbc.type == FaceBC::Neumann) {
                            rhs += fbc.value(L_, y, z) * faceArea;
                        } else if (fbc.type == FaceBC::Robin) {
                            const double u_bound = faceArea / ((dx_ / (2.0 * g_c)) + (1.0 / fbc.h));
                            diag += u_bound;
                            rhs += u_bound * fbc.value(L_, y, z);
                        }
                    }

                    // --- Y-Axis Neighbors ---
                    if (j > 0) {
                        const std::size_t m = idx(i, j - 1, k);
                        const double g_face = dx_ * (2.0 * g_c * gamma[m]) / (g_c + gamma[m]);
                        diag += g_face;
                        rhs += g_face * phi[m];
                    } else {
                        // YLow Boundary
                        const auto& fbc = bc[YLow];
                        if (fbc.type == FaceBC::Dirichlet) {
                            const double g_bound = 2.0 * g_c * dx_;
                            diag += g_bound;
                            rhs += g_bound * fbc.value(x, 0.0, z);
                        } else if (fbc.type == FaceBC::Neumann) {
                            rhs += fbc.value(x, 0.0, z) * faceArea;
                        } else if (fbc.type == FaceBC::Robin) {
                            const double u_bound = faceArea / ((dx_ / (2.0 * g_c)) + (1.0 / fbc.h));
                            diag += u_bound;
                            rhs += u_bound * fbc.value(x, 0.0, z);
                        }
                    }

                    if (j + 1 < n_) {
                        const std::size_t m = idx(i, j + 1, k);
                        const double g_face = dx_ * (2.0 * g_c * gamma[m]) / (g_c + gamma[m]);
                        diag += g_face;
                        rhs += g_face * phi[m];
                    } else {
                        // YHigh Boundary
                        const auto& fbc = bc[YHigh];
                        if (fbc.type == FaceBC::Dirichlet) {
                            const double g_bound = 2.0 * g_c * dx_;
                            diag += g_bound;
                            rhs += g_bound * fbc.value(x, L_, z);
                        } else if (fbc.type == FaceBC::Neumann) {
                            rhs += fbc.value(x, L_, z) * faceArea;
                        } else if (fbc.type == FaceBC::Robin) {
                            const double u_bound = faceArea / ((dx_ / (2.0 * g_c)) + (1.0 / fbc.h));
                            diag += u_bound;
                            rhs += u_bound * fbc.value(x, L_, z);
                        }
                    }

                    // --- Z-Axis Neighbors ---
                    if (k > 0) {
                        const std::size_t m = idx(i, j, k - 1);
                        const double g_face = dx_ * (2.0 * g_c * gamma[m]) / (g_c + gamma[m]);
                        diag += g_face;
                        rhs += g_face * phi[m];
                    } else {
                        // ZLow Boundary
                        const auto& fbc = bc[ZLow];
                        if (fbc.type == FaceBC::Dirichlet) {
                            const double g_bound = 2.0 * g_c * dx_;
                            diag += g_bound;
                            rhs += g_bound * fbc.value(x, y, 0.0);
                        } else if (fbc.type == FaceBC::Neumann) {
                            rhs += fbc.value(x, y, 0.0) * faceArea;
                        } else if (fbc.type == FaceBC::Robin) {
                            const double u_bound = faceArea / ((dx_ / (2.0 * g_c)) + (1.0 / fbc.h));
                            diag += u_bound;
                            rhs += u_bound * fbc.value(x, y, 0.0);
                        }
                    }

                    if (k + 1 < n_) {
                        const std::size_t m = idx(i, j, k + 1);
                        const double g_face = dx_ * (2.0 * g_c * gamma[m]) / (g_c + gamma[m]);
                        diag += g_face;
                        rhs += g_face * phi[m];
                    } else {
                        // ZHigh Boundary
                        const auto& fbc = bc[ZHigh];
                        if (fbc.type == FaceBC::Dirichlet) {
                            const double g_bound = 2.0 * g_c * dx_;
                            diag += g_bound;
                            rhs += g_bound * fbc.value(x, y, L_);
                        } else if (fbc.type == FaceBC::Neumann) {
                            rhs += fbc.value(x, y, L_) * faceArea;
                        } else if (fbc.type == FaceBC::Robin) {
                            const double u_bound = faceArea / ((dx_ / (2.0 * g_c)) + (1.0 / fbc.h));
                            diag += u_bound;
                            rhs += u_bound * fbc.value(x, y, L_);
                        }
                    }

                    // SOR Update
                    if (diag == 0.0) {
                        throw std::runtime_error("Singular matrix encountered: cell diagonal is zero.");
                    }
                    const double phi_new_gauss = rhs / diag;
                    const double delta = omega * (phi_new_gauss - phi[c]);
                    phi[c] += delta;

                    maxDelta = std::max(maxDelta, std::abs(delta));
                }
            }
        }

        if (maxDelta < tol) {
            return true;
        }
    }
    return false;
}

void ElectroThermal3D::computeJoule(const std::vector<double>& sigma) {
    const double cellVolume = dx_ * dx_ * dx_;

    for (std::size_t i = 0; i < n_; ++i) {
        const double x = center(i);
        for (std::size_t j = 0; j < n_; ++j) {
            const double y = center(j);
            for (std::size_t k = 0; k < n_; ++k) {
                const double z = center(k);

                const std::size_t c = idx(i, j, k);
                const double sig_c = sigma[c];
                const double V_c = V_[c];
                double power_cell = 0.0;

                // --- X Neighbors ---
                if (i > 0) {
                    const std::size_t m = idx(i - 1, j, k);
                    const double g_face = dx_ * (2.0 * sig_c * sigma[m]) / (sig_c + sigma[m]);
                    const double dV = V_c - V_[m];
                    const double p_branch = g_face * dV * dV;
                    // Voltage divider: fraction of resistance inside cell c is sigma_m / (sig_c + sigma_m)
                    power_cell += p_branch * (sigma[m] / (sig_c + sigma[m]));
                } else if (ebc_[XLow].type == FaceBC::Dirichlet) {
                    const double g_bound = 2.0 * sig_c * dx_;
                    const double dV = V_c - ebc_[XLow].value(0.0, y, z);
                    power_cell += g_bound * dV * dV; // Entire half-cell resistor resides inside cell c
                }

                if (i + 1 < n_) {
                    const std::size_t m = idx(i + 1, j, k);
                    const double g_face = dx_ * (2.0 * sig_c * sigma[m]) / (sig_c + sigma[m]);
                    const double dV = V_c - V_[m];
                    const double p_branch = g_face * dV * dV;
                    power_cell += p_branch * (sigma[m] / (sig_c + sigma[m]));
                } else if (ebc_[XHigh].type == FaceBC::Dirichlet) {
                    const double g_bound = 2.0 * sig_c * dx_;
                    const double dV = V_c - ebc_[XHigh].value(L_, y, z);
                    power_cell += g_bound * dV * dV;
                }

                // --- Y Neighbors ---
                if (j > 0) {
                    const std::size_t m = idx(i, j - 1, k);
                    const double g_face = dx_ * (2.0 * sig_c * sigma[m]) / (sig_c + sigma[m]);
                    const double dV = V_c - V_[m];
                    const double p_branch = g_face * dV * dV;
                    power_cell += p_branch * (sigma[m] / (sig_c + sigma[m]));
                } else if (ebc_[YLow].type == FaceBC::Dirichlet) {
                    const double g_bound = 2.0 * sig_c * dx_;
                    const double dV = V_c - ebc_[YLow].value(x, 0.0, z);
                    power_cell += g_bound * dV * dV;
                }

                if (j + 1 < n_) {
                    const std::size_t m = idx(i, j + 1, k);
                    const double g_face = dx_ * (2.0 * sig_c * sigma[m]) / (sig_c + sigma[m]);
                    const double dV = V_c - V_[m];
                    const double p_branch = g_face * dV * dV;
                    power_cell += p_branch * (sigma[m] / (sig_c + sigma[m]));
                } else if (ebc_[YHigh].type == FaceBC::Dirichlet) {
                    const double g_bound = 2.0 * sig_c * dx_;
                    const double dV = V_c - ebc_[YHigh].value(x, L_, z);
                    power_cell += g_bound * dV * dV;
                }

                // --- Z Neighbors ---
                if (k > 0) {
                    const std::size_t m = idx(i, j, k - 1);
                    const double g_face = dx_ * (2.0 * sig_c * sigma[m]) / (sig_c + sigma[m]);
                    const double dV = V_c - V_[m];
                    const double p_branch = g_face * dV * dV;
                    power_cell += p_branch * (sigma[m] / (sig_c + sigma[m]));
                } else if (ebc_[ZLow].type == FaceBC::Dirichlet) {
                    const double g_bound = 2.0 * sig_c * dx_;
                    const double dV = V_c - ebc_[ZLow].value(x, y, 0.0);
                    power_cell += g_bound * dV * dV;
                }

                if (k + 1 < n_) {
                    const std::size_t m = idx(i, j, k + 1);
                    const double g_face = dx_ * (2.0 * sig_c * sigma[m]) / (sig_c + sigma[m]);
                    const double dV = V_c - V_[m];
                    const double p_branch = g_face * dV * dV;
                    power_cell += p_branch * (sigma[m] / (sig_c + sigma[m]));
                } else if (ebc_[ZHigh].type == FaceBC::Dirichlet) {
                    const double g_bound = 2.0 * sig_c * dx_;
                    const double dV = V_c - ebc_[ZHigh].value(x, y, L_);
                    power_cell += g_bound * dV * dV;
                }

                // Volumetric Joule density q_v [W/m^3]
                qv_[c] = power_cell / cellVolume;
            }
        }
    }
}

double ElectroThermal3D::voltage(std::size_t i, std::size_t j, std::size_t k) const {
    return V_[idx(i, j, k)];
}

double ElectroThermal3D::jouleDensity(std::size_t i, std::size_t j, std::size_t k) const {
    return qv_[idx(i, j, k)];
}

double ElectroThermal3D::totalJoulePower() const {
    const double cellVolume = dx_ * dx_ * dx_;
    double totalPower = 0.0;
    for (double q : qv_) {
        totalPower += q * cellVolume;
    }
    return totalPower;
}

double ElectroThermal3D::center(std::size_t i) const {
    return (static_cast<double>(i) + 0.5) * dx_;
}

} // namespace physi_sim::thermal3d
