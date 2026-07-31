// src/apps/export_chip_stack.cpp
//
// Reads a declarative package layout (JSON), samples it onto the solver grid,
// and exports material / conductivity / power fields to .vti for ParaView.
//
// COMPOSITION, not coupling: this app links BOTH io_component (layout parsing,
// VTK export) and thermal3d (physics). Neither component depends on the other --
// thermal3d stays at zero dependencies and io keeps nlohmann_json PRIVATE. The
// app is where they meet, and plain std::vector<double> is the only currency
// crossing between them.
//
// SCOPE NOTE. Geometry export works for any domain. The steady solve is gated on
// a CUBIC domain because ElectroThermal3D currently takes a single n and a single
// L. A realistic 2.5D package is ~6 mm laterally and ~0.5 mm tall -- 12:1 -- and
// that aspect ratio is not incidental: the lateral healing length of the
// interposer is lambda = sqrt(k t R'') ~ 670 um, so GPU->HBM crosstalk only
// resolves when the domain spans several lambda. Until the solver accepts
// (nx,ny,nz) and (Lx,Ly,Lz), use python/scripts/solve_layout_reference.py, which
// solves the same JSON anisotropically and will serve as the cross-implementation
// check once the C++ path lands.

#include "io/ChipLayout.hpp"
#include "io/VTKWriter.hpp"
#include "thermal3d/ElectroThermal3D.hpp"

#include <cstdio>
#include <string>
#include <vector>

using physi_sim::io::ChipLayout;
using physi_sim::io::VTKWriter;
using physi_sim::io::loadChipLayout;
using physi_sim::thermal3d::ElectroThermal3D;

int main(int argc, char* argv[]) {
    const std::string path = (argc > 1) ? argv[1] : "chip_layout.json";

    ChipLayout layout;
    try {
        layout = loadChipLayout(path);          // throws with a specific message
    } catch (const std::exception& e) {
        std::printf("FATAL: %s\n", e.what());
        return 1;                                // fail loudly: a silent fallback to
    }                                            // hardcoded geometry is how a config
                                                 // file becomes decorative
    const auto n = layout.cells;
    const auto h = layout.spacing();
    const std::size_t total = n[0] * n[1] * n[2];

    std::printf("layout   : %s\n", path.c_str());
    std::printf("domain   : %.0f x %.0f x %.0f um   (aspect %.1f:1)\n",
                layout.size[0]*1e6, layout.size[1]*1e6, layout.size[2]*1e6,
                layout.size[0]/layout.size[2]);
    std::printf("grid     : %zu x %zu x %zu = %zu cells   dx=%.1f dy=%.1f dz=%.1f um\n",
                n[0], n[1], n[2], total, h[0]*1e6, h[1]*1e6, h[2]*1e6);

    // ---- sample the layout onto cell centres --------------------------------
    // Cell CENTRES, matching the finite-volume convention: each unknown owns a box
    // of one material, and conductivity jumps only at faces. That piecewise-constant
    // model is exactly what makes the harmonic-mean face conductivity exact at an
    // interface rather than merely second-order.
    std::vector<double> matId(total), kField(total), qField(total);
    std::vector<std::string> ids;                 // region name -> numeric id for ParaView

    for (std::size_t i = 0; i < n[0]; ++i)
        for (std::size_t j = 0; j < n[1]; ++j)
            for (std::size_t l = 0; l < n[2]; ++l) {
                const double x = (static_cast<double>(i) + 0.5) * h[0];
                const double y = (static_cast<double>(j) + 0.5) * h[1];
                const double z = (static_cast<double>(l) + 0.5) * h[2];
                const std::size_t p = (i * n[1] + j) * n[2] + l;

                const auto* r = layout.regionAt(x, y, z);   // loader already proved non-null
                std::size_t id = 0;
                for (; id < ids.size(); ++id) if (ids[id] == r->name) break;
                if (id == ids.size()) ids.push_back(r->name);

                matId[p]  = static_cast<double>(id + 1);
                kField[p] = layout.kAt(x, y, z);
                qField[p] = layout.sourceAt(x, y, z);
            }

    std::printf("regions  :");
    for (std::size_t i = 0; i < ids.size(); ++i) std::printf(" %zu=%s", i+1, ids[i].c_str());
    std::printf("\n");

    // ---- export geometry ----------------------------------------------------
    // Three fields, honestly named. `material_id` is a LABEL, not a temperature:
    // exporting an ID map under a temperature name produces a plausible-looking
    // ParaView render of nothing.
    VTKWriter w;
    const bool ok = w.write_3d(matId, "material_id", n[0], n[1], n[2], h[0], h[1], h[2],
                               "chip_layout.vti");
    if (!ok) { std::printf("FATAL: could not write chip_layout.vti\n"); return 1; }
    w.write_3d(kField, "k_w_mk",     n[0], n[1], n[2], h[0], h[1], h[2], "chip_k.vti");
    w.write_3d(qField, "q_v_w_m3",   n[0], n[1], n[2], h[0], h[1], h[2], "chip_power.vti");
    std::printf("wrote    : chip_layout.vti, chip_k.vti, chip_power.vti\n");

    // ---- solve, if the current solver can represent this domain -------------
    const bool cubic = (n[0] == n[1] && n[1] == n[2]) &&
                       (layout.size[0] == layout.size[1] && layout.size[1] == layout.size[2]);
    if (!cubic) {
        std::printf("\nsolve    : SKIPPED -- ElectroThermal3D is cubic-only (single n, single L).\n"
                    "           This domain is %.1f:1. Run the anisotropic reference:\n"
                    "             python3 python/scripts/solve_layout_reference.py %s\n",
                    layout.size[0]/layout.size[2], path.c_str());
        return 0;
    }

    ElectroThermal3D solver(n[0], layout.size[0], [](double,double,double){ return 1.0; });
    ElectroThermal3D::BCs bcs;
    for (auto& f : bcs) { f.type = ElectroThermal3D::FaceBC::Neumann;
                          f.value = [](double,double,double){ return 0.0; }; }
    const auto zh = layout.boundaries.find("z_high");
    if (zh != layout.boundaries.end() && zh->second.type == physi_sim::io::FaceBoundary::Type::Robin) {
        const double hFilm = zh->second.h, tInf = zh->second.tInfK;
        bcs[ElectroThermal3D::ZHigh].type  = ElectroThermal3D::FaceBC::Robin;
        bcs[ElectroThermal3D::ZHigh].h     = hFilm;
        bcs[ElectroThermal3D::ZHigh].value = [tInf](double,double,double){ return tInf; };
    }

    std::vector<double> T(total, 300.0);
    const auto& lay = layout;
    if (!solver.solveThermal([&lay](double x,double y,double z){ return lay.kAt(x,y,z); },
                             bcs, qField, T, 1e-9)) {
        std::printf("solve    : DID NOT CONVERGE\n");
        return 1;
    }

    std::vector<double> Tc(total);
    double tmin = 1e9, tmax = -1e9;
    for (std::size_t p = 0; p < total; ++p) { Tc[p] = T[p] - 273.15;
        tmin = std::min(tmin, Tc[p]); tmax = std::max(tmax, Tc[p]); }
    w.write_3d(Tc, "temperature_C", n[0], n[1], n[2], h[0], h[1], h[2], "chip_stack.vti");
    std::printf("solve    : converged   T %.2f .. %.2f C   -> chip_stack.vti\n", tmin, tmax);
    return 0;
}
