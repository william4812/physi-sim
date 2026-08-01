#pragma once
//
// io/ChipLayout.hpp
//
// Declarative package geometry: a JSON-described stack of materials and regions
// that a solver can sample onto any grid.
//
// EXTENSIBILITY IS THE POINT. Material properties live in an open key/value map,
// not in fixed struct fields. Adding a property -- rho_kg_m3 and cp_j_kgk for the
// transient solve, tau_phonon_s for the sub-continuum bridge, k_xx/k_zz for
// anisotropic HBM stacks -- requires editing the JSON and nothing else. No struct
// change, no parser change, no recompile of consumers that do not use it. Named
// accessors exist only for the two properties every solve needs (k and sigma);
// everything else is read with get("key", fallback).
//
// DEPENDENCY DISCIPLINE. This header includes no JSON library. The struct is
// plain data, so it can cross any component boundary. Parsing lives in
// ChipLayout.cpp, which is compiled into io_component where nlohmann_json is
// already linked PRIVATE. thermal3d therefore stays at zero dependencies: an app
// links both components and passes plain vectors between them.
//
// UNITS. The JSON is authored in engineering units (um, W/cm^2, C) because that
// is how packages are specified. Everything in this struct is SI (m, W/m^3, K)
// because that is what the solver consumes. The conversion happens once, in the
// loader, so no downstream code has to remember which is which.

#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace physi_sim::io {

/// A material's properties, held as an open map so new physics needs no code change.
struct Material {
    std::string name;
    std::map<std::string, double> props;   ///< SI units, keys as authored in JSON

    /// Property or `fallback` if absent. Use for optional physics.
    double get(const std::string& key, double fallback) const;
    /// Property or throw with a message naming the material and key.
    /// Use for physics the solve cannot proceed without.
    double require(const std::string& key) const;

    double k()     const { return require("k_w_mk");        }  ///< W/(m K)
    double sigma() const { return get("sigma_s_m", 0.0);    }  ///< S/m
    double tLimitC() const { return get("t_limit_c", 1.0e9); } ///< C, for margin reporting
};

/// An axis-aligned box of one material. Unset bounds span the whole domain.
struct Region {
    std::string name;
    std::string material;                          ///< key into ChipLayout::materials
    std::optional<std::array<double, 2>> x, y, z;  ///< [lo, hi) in METRES
    double powerSurfaceWm2 = 0.0;                  ///< W/m^2 over the region footprint

    bool contains(double px, double py, double pz) const;

    /// Volumetric source [W/m^3]: surface power divided by the region's z-thickness.
    /// Power is a property of the REGION, not the material, because the same
    /// silicon can be a powered die in one region and a dummy stiffener in another,
    /// and because the conversion needs a thickness the material does not know.
    double volumetricSource() const;
};

/// One face of the domain.
struct FaceBoundary {
    enum class Type { Adiabatic, Robin, Dirichlet };
    Type   type  = Type::Adiabatic;
    double h     = 0.0;      ///< W/(m^2 K), Robin only
    double tInfK = 0.0;      ///< K, Robin ambient or Dirichlet value
};

/// The whole package description.
struct ChipLayout {
    std::array<double, 3>      size{};    ///< Lx, Ly, Lz [m]
    std::array<std::size_t, 3> cells{};   ///< nx, ny, nz
    std::map<std::string, Material> materials;
    std::vector<Region>             regions;   ///< APPLIED IN ORDER, later wins
    std::map<std::string, FaceBoundary> boundaries;  ///< keys: x_low..z_high

    /// Cell size along each axis [m].
    std::array<double, 3> spacing() const;

    /// The last region containing the point, or nullptr. LAST-WINS is deliberate:
    /// it lets a full-width region act as the fill for a z-band and a narrower
    /// region placed after it carve out a feature -- which is how a GPU die is cut
    /// out of a mold-compound layer with HBM on either side. It also means region
    /// ORDER IN THE JSON IS SEMANTIC, not cosmetic.
    const Region* regionAt(double x, double y, double z) const;

    /// Thermal conductivity [W/(m K)] at a point. Throws if no region covers it,
    /// because a hole in the layout is an authoring error, not a default.
    double kAt(double x, double y, double z) const;

    /// Volumetric heat source [W/m^3] at a point; zero where no region has power.
    double sourceAt(double x, double y, double z) const;

    /// Material at a point, for reporting limits and labelling exports.
    const Material& materialAt(double x, double y, double z) const;
};

/// Parse a layout file. Converts authoring units (um, W/cm^2, C) to SI.
/// @throws std::runtime_error on missing file, malformed JSON, unknown material
///         reference, or a region whose bounds fall outside the domain.
ChipLayout loadChipLayout(const std::string& path);

}  // namespace physi_sim::io
