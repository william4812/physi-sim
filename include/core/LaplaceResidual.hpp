// include/core/LaplaceResidual.hpp
//
// The UNIFIED residual measure for the 2D steady-state heat problem.
//
// WHAT IT IS
//   For the discretized Laplace equation A·T = b (here b = 0, no heat source),
//   the "equation residual" measures how far a field T is from actually
//   SOLVING the equation, at every interior node:
//
//       r_ij = | T(i+1,j) + T(i-1,j) + T(i,j+1) + T(i,j-1) - 4·T(i,j) |
//
//   and we report the worst node:  || r ||_inf = max_ij r_ij  (in Kelvin).
//   When this is ~0, every interior node equals the average of its four
//   neighbours — which is exactly what steady-state conduction requires.
//
// WHY IT LIVES HERE AND NOT INSIDE A SOLVER
//   This function depends ONLY on core::Grid2D. It does not know about Jacobi,
//   TDMA, the GPU, or the iteration loop — it just reads a field and returns a
//   number. That is the point: there is ONE definition of "residual", and
//   every solver is judged by the same yardstick. Iteration is the solver's
//   job; measuring is this function's job. Keeping them apart is what makes
//   the comparison across algorithms fair (and the code low-coupling).
//
// STATELESS
//   No members, no lifecycle, side-effect-free. Header-only and inline, so it
//   adds no translation unit and needs no CMake change — just #include it.
//
// RELATION TO THE INCREMENT RESIDUAL EACH SOLVER ALREADY REPORTS
//   The solvers' existing residual() is the *increment* || T^k - T^{k-1} ||_inf.
//   For Jacobi specifically, the update is T^{k+1} = ¼·(sum of 4 neighbours),
//   so  increment_ij = ¼·r_ij  exactly. Hence for a Jacobi step:
//       equation_residual = 4 × (Jacobi increment).
//   The unit test uses this exact identity to prove this function is correct.
//
// FUTURE EXTENSION
//   If a heat source b_ij is ever added, subtract it here:
//       r_ij = | (4-point sum) - 4·T(i,j) - b_ij |.
//   Boundary nodes are Dirichlet (fixed) and satisfied by construction, so the
//   loop intentionally covers interior nodes only.

#pragma once

#include "core/Grid2D.hpp"
#include <algorithm>   // std::max
#include <cmath>       // std::abs

namespace physi_sim::core {

[[nodiscard]] inline double laplace_residual_linf(const Grid2D& field)
{
    const int nx = field.get_nx();
    const int ny = field.get_ny();

    double max_res = 0.0;

    // Interior nodes only: i in [1, nx-2], j in [1, ny-2].
    for (int j = 1; j < ny - 1; ++j)
    {
        for (int i = 1; i < nx - 1; ++i)
        {
            const double laplacian =
                  field.at(i + 1, j) + field.at(i - 1, j)
                + field.at(i, j + 1) + field.at(i, j - 1)
                - 4.0 * field.at(i, j);

            max_res = std::max(max_res, std::abs(laplacian));
        }
    }

    return max_res;   // 0.0 for a 1×1 / 2×2 grid with no interior — safe.
}

} // namespace physi_sim::core
