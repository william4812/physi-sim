#pragma once
#include "core/Grid2D.hpp"
#include <string>

/**
 * @file ISolver.hpp
 * @brief Abstract solver interface — Strategy Pattern over existing Grid2D.
 *
 * FIRST PRINCIPLE — no duplication:
 *   Uses physi_sim::core::Grid2D directly — the existing, tested grid type.
 *   No new Grid struct. No parallel class hierarchy.
 *   ISolver is a thin abstraction layer over what already works.
 *
 * FIRST PRINCIPLE — namespace consistency:
 *   Lives in physi_sim::solver to match the existing physi_sim::core pattern.
 *   Clear ownership: core owns data types, solver owns algorithms.
 *
 * ANDURIL/NVIDIA INTERVIEW:
 *   "I wrapped the existing Grid2D rather than duplicating it — single source
 *   of truth for the grid data structure. Adding a new backend never requires
 *   touching Grid2D."
 */

namespace physi_sim::solver {

class ISolver {
public:
    virtual ~ISolver() = default;

    /**
     * @brief One iteration of the solver on grid.
     *
     * Uses existing Grid2D interface — operator(), get_nx(), get_ny().
     * Boundary values in the halo (index 0 and nx-1/ny-1) are read
     * but never modified — caller sets boundary conditions once before
     * the solve loop.
     */
    virtual void        step(core::Grid2D& grid) = 0;

    /**
     * @brief L-inf residual after the most recent step().
     *
     * max|T_new - T_old| over all interior points.
     * This matches the convergence criterion already used in Solver2D
     * and verified in README §2 (bound 5e-4).
     */
    virtual double      residual() const = 0;

    /**
     * @brief Human-readable name for logging and CSV headers.
     * Examples: "JacobiCPU", "TDMACPU", "JacobiGPU"
     */
    virtual std::string name()     const = 0;

    virtual const std::vector<double>& get_history() const = 0;
};

enum class HardwareBackend { CPU, CUDA };
enum class SolverType      { JACOBI, TDMA };

} // namespace physi_sim::solver
