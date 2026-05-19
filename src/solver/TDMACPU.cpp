// src/solver/TDMACPU.cpp
#include "solver/TDMACPU.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

namespace physi_sim::solver {

void TDMACPU::step(core::Grid2D& grid) {
    const int nx = grid.get_nx();
    const int ny = grid.get_ny();
    residual_ = 0.0;
    std::vector<double> a(ny), b(ny), c(ny), d(ny), T_new(ny);

    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            a[j] = -1.0;
            b[j] =  4.0;
            c[j] = -1.0;
            d[j] = grid(i-1, j) + grid(i+1, j);
        }
        // ── Add Dirichlet boundary contributions to RHS ───────────────────────
    // The tridiagonal is defined only for interior rows j=1..ny-2.
    // The boundary rows j=0 (bottom) and j=ny-1 (top) are known values.
    // They appear as c[j]*T[j+1] and a[j]*T[j-1] terms — when T is known,
    // move to RHS and negate: d[j] -= coeff * T_boundary.
    // Since a[1] = -1 and c[ny-2] = -1:  -(-1)*value = +value.
        d[1]      += grid(i, 0);        // ← bottom BC: row j=0 feeds into j=1
        d[ny - 2] += grid(i, ny - 1);  // ← top BC:    row j=ny-1 feeds into j=ny-2
        
        for (int j = 2; j < ny - 1; ++j) {
            double m = a[j] / b[j-1];
            b[j] -= m * c[j-1];
            d[j] -= m * d[j-1];
        }
        T_new[ny-2] = d[ny-2] / b[ny-2];
        for (int j = ny - 3; j >= 1; --j)
            T_new[j] = (d[j] - c[j] * T_new[j+1]) / b[j];
        for (int j = 1; j < ny - 1; ++j) {
            residual_ = std::max(residual_, std::abs(T_new[j] - grid(i, j)));
            grid(i, j) = T_new[j];
        }
    }
}

double      TDMACPU::residual() const { return residual_; }
std::string TDMACPU::name()     const { return "TDMACPU"; }

} // namespace physi_sim::solver
