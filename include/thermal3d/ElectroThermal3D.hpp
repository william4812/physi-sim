#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

namespace physi_sim::thermal3d 
{

/**
 * @file ElectroThermal3D.hpp
 * @brief Steady electro-thermal solver for a 3D TSV / microbump unit cell.
 *
 * PHYSICAL REGIME
 * ---------------------------------------------------------------------------
 * The verification domain is a unit cell of on-package DC power delivery: a
 * high-conductivity copper interconnect (TSV, microbump, RDL trace) embedded in
 * a far less conductive silicon / dielectric matrix, as found in 2.5D and 3D
 * stacks (CoWoS-style interposers, HBM stacks, stacked-cache dies).
 *
 * Current is forced through the cell by a potential difference across two
 * opposing faces; the remaining faces are insulated, so the cell represents one
 * repeating element of a via array.
 *
 * PHYSICS - two fields, one operator
 * ---------------------------------------------------------------------------
 * Charge conservation at DC (div J = 0) with Ohm's law (J = -sigma grad V), and
 * energy conservation with Fourier's law (q = -k grad T), are THE SAME elliptic
 * operator with different coefficients:
 *
 *      ELECTRICAL   div(sigma grad V) = 0
 *      JOULE        q_v = sigma |grad V|^2        [W/m^3]   (this is J . E)
 *      THERMAL      div(k grad T) + q_v = 0
 *
 * so one assembly engine serves both - pass sigma for charge, k for heat. The
 * coupling is one-way: V -> q_v -> T. The thermal source cannot be assembled
 * until V is known, which FORCES the solve order. (Two-way coupling via
 * sigma(T) - the runaway loop where heating raises resistivity raises heating -
 * needs an outer Picard iteration and is deliberately not implemented.)
 *
 * CURRENT CROWDING
 * ---------------------------------------------------------------------------
 * Dissipation goes as the SQUARE of the potential gradient, so wherever
 * geometry or a conductivity contrast forces current into a smaller area,
 * |grad V| rises and q_v rises quadratically. Via necks, bump-to-pad
 * transitions and RDL corners are therefore electro-thermal hotspots, not
 * merely electrical ones. The assembly must not smear these spikes: face
 * conductances use the harmonic mean of the two adjacent cells, which is exact
 * at a conductivity discontinuity, so a Cu|dielectric boundary stays sharp
 * instead of being averaged into a fictitious intermediate material.
 *
 * SCOPE - Cycle 2a
 * ---------------------------------------------------------------------------
 * This cycle implements the ELECTRICAL half only: solve V, derive q_v, and pin
 * the volume integral of q_v against the lumped circuit result. The therml
 * solve (Cycle 2b) reuses the same private engine and is added as an additive,
 * non-breaking extension - no method here changes signature.
 *
 * DESIGN - why this class does not implement ISolver
 * ---------------------------------------------------------------------------
 * ISolver is, by its own header comment, "a Strategy Pattern over existing
 * Grid2D" and includes core/Grid2D.hpp. It is structurally 2D. Implementing it
 * from a 3D solver would require widening ISolver, touching all four existing
 * solvers, the factory and the FSM tests. ElectroThermal3D therefore stands
 * alone in thermal3d, as the verified Conduction3D already does, and composes
 * rather than inherits. Once both 3D solvers exist an ISpatialSolver interface
 * is EARNED - two implementations, not one - and is extracted in its own
 * dedicated refactor cycle.
 */
class ElectroThermal3D {
public:
    /// Continuous material or boundary field sampled at a point (x, y, z) in metres.
    using ScalarField = std::function<double(double, double, double)>;

    /// Face index into the BCs array. Named so tests read as physics rather
    /// than magic numbers: bcs[ElectroThermal3D::ZLow], not bcs[4].
    enum Face : std::size_t { XLow = 0, XHigh, YLow, YHigh, ZLow, ZHigh, FaceCount };

    /**
     * @brief Boundary condition on one face of the cell.
     *
     * All three types are the same series-resistance idea seen from different
     * limits: a boundary is a face whose neighbour is not a cell.
     *
     *   Dirichlet - `value` is the fixed potential [V] at the face. Conductance
     *               is the HALF-CELL path only, sigma * A / (dx/2), because the
     *               cell centre sits dx/2 from the face. Physically this is a
     *               perfect ohmic contact: the electrode imposes its potential
     *               directly on the face with no interfacial resistance.
     *               Using the full-cell value instead silently inserts a
     *               fictitious contact resistance at the electrode.
     *   Neumann   - `value` is the INWARD current density [A/m^2] through the
     *               face. Zero means insulated, which is the symmetry condition
     *               on the lateral walls of a repeating via-array unit cell.
     *               Contributes to the RHS only; the diagonal is untouched,
     *               because no conductance path to an exterior node exists.
     *   Robin     - `value` is the ambient, `h` the film coefficient. Present
     *               for the thermal half (Cycle 2b): half-cell conduction in
     *               series with a convective film, U = 1/(dx/(2*gamma) + 1/h).
     */
    struct FaceBC {
        enum Type { Dirichlet, Neumann, Robin };

        Type        type  = Neumann;                                    ///< default: insulated
        ScalarField value = [](double, double, double) { return 0.0; }; ///< sampled at face centre
        double      h     = 0.0;                                        ///< film coefficient; Robin only
    };

    /// Boundary conditions for all six faces, indexed by Face.
    using BCs = std::array<FaceBC, FaceCount>;

    /**
     * @brief Construct a solver on a uniform cubic grid of n^3 cells.
     * @param n       cells per axis; must be >= 2.
     * @param length  cell edge length [m]; must be > 0.
     * @param sigma   electrical conductivity field [S/m]; must be > 0 everywhere.
     * @throws std::invalid_argument if n < 2 or length <= 0.
     *
     * sigma is sampled once at cell CENTRES during solve(). Sampling at centres
     * rather than faces is precisely what makes the harmonic mean the correct
     * face conductivity - see the derivation in assemble().
     */
    ElectroThermal3D(std::size_t n, double length, ScalarField sigma);

    /**
     * @brief Set the six electrical boundary conditions.
     *
     * Must be called before solve(). The default (all six insulated) leaves the
     * problem singular - a floating conductor has no reference potential - so at
     * least one face must be Dirichlet.
     */
    void setElectricalBC(const BCs& bc);

    /**
     * @brief Solve div(sigma grad V) = 0, then derive q_v = sigma |grad V|^2.
     * @param tol      convergence tolerance on the maximum cell update.
     * @param maxIter  iteration cap; reaching it returns false.
     * @return true if the field solve converged below tol.
     */
    bool solve(double tol = 1e-12, std::size_t maxIter = 500000);

    bool solveThermal(const ScalarField& k_field, const BCs& bcs, 
                      const std::vector<double>& src, std::vector<double>& T, 
                      double tol);
    /// Electric potential [V] at cell (i, j, k). Valid after solve().
    double voltage(std::size_t i, std::size_t j, std::size_t k) const;

    /// Volumetric Joule dissipation sigma*|grad V|^2 [W/m^3] at cell (i, j, k).
    /// Valid after solve(). This is the term that becomes the thermal source.
    double jouleDensity(std::size_t i, std::size_t j, std::size_t k) const;

    /**
     * @brief Total dissipated electrical power [W]: the volume integral of
     *        sigma*|grad V|^2 over the cell.
     *
     * THE VERIFICATION PIN. For a uniform conductor this must equal the lumped
     * circuit result I^2 R, because the 7-point stencil is exact for the linear
     * potential field such a problem produces - so the residual error is
     * grid-independent and at machine precision, not O(dx^2).
     *
     * Equating a field-theoretic volume integral with a circuit law is the First
     * Law of Thermodynamics stated numerically: every joule of electrical work
     * delivered across the chip-to-interposer boundary emerges as heat inside
     * the cell, no more and no less.
     */
    double totalJoulePower() const;

    /// Centre coordinate [m] of index i along any axis: (i + 0.5) * dx.
    double center(std::size_t i) const;

    /// Cells per axis.
    std::size_t n() const { return n_; }

    /// Uniform cell size [m].
    double dx() const { return dx_; }

private:
    /// Row-major flattening: (i * n + j) * n + k. Matches Conduction3D so that
    /// a future shared kernel sees one memory layout, not two.
    std::size_t idx(std::size_t i, std::size_t j, std::size_t k) const {
        return (i * n_ + j) * n_ + k;
    }

    /// Sample a continuous material field at every cell centre.
    std::vector<double> sampleCells(const ScalarField& f) const;

    /// Assemble the 7-point stencil for div(gamma grad phi) + src = 0 under
    /// per-face BCs, then SOR-solve it. Shared by the electrical problem
    /// (gamma = sigma) and, in Cycle 2b, the thermal one (gamma = k).
    bool solveField(const std::vector<double>& gamma, const BCs& bc,
                    const std::vector<double>& src, std::vector<double>& phi,
                    double tol, std::size_t maxIter);

    /// Fill qv_ with sigma*|grad V|^2 from the converged potential field.
    void computeJoule(const std::vector<double>& sigma);

    std::size_t n_;
    double      L_;
    double      dx_;
    ScalarField sigmaField_;
    BCs         ebc_;
    std::vector<double> V_;   ///< electric potential [V]
    std::vector<double> qv_;  ///< Joule dissipation density [W/m^3]
};

}  // namespace physi_sim::thermal3d
