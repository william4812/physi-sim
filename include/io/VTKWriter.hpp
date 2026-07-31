#pragma once
#include <vector>
#include <string>
#include <cstddef>

namespace physi_sim::io {

/**
 * @brief Writes simulation fields to VTK format for visualisation.
 *
 * Standalone — no interface inheritance needed until a second
 * writer implementation exists (Open/Closed: don't abstract prematurely).
 */
class VTKWriter {
public:
    VTKWriter() = default;
    virtual ~VTKWriter() = default;

    void write(const std::vector<double>& field,
               const std::string& filename);

    void write_2d(const double* field,
                  int nx, int ny,
                  const std::string& filename);

    /**
     * @brief Write a 3D cell-centred field to XML VTImageData (.vti), ParaView-native.
     *
     * Uses CellData rather than PointData because a finite-volume unknown is the
     * CELL AVERAGE over its control volume; interpolating to points would blur
     * the gradient jump at a material interface (e.g. Si|TIM). An optional vector
     * field (heat flux, current density) is written alongside the scalar.
     *
     * INDEX ORDER: inputs use the solver flattening (i*ny + j)*nz + k (k fastest).
     * VTK serialises x fastest, so this method reorders on write; a transposed
     * writer still renders but is silently wrong, so the ordering is pinned by
     * tests/unit/test_field_export_vti.cpp.
     *
     * ANISOTROPIC by design: a package is ~6 mm laterally and ~0.4 mm tall, so a
     * cubic grid cannot represent one. The cubic overload below delegates here.
     *
     * @return false if the file cannot be opened.
     */
    bool write_3d(const std::vector<double>& scalar,
                  const std::string& scalarName,
                  std::size_t nx, std::size_t ny, std::size_t nz,
                  double dx, double dy, double dz,
                  const std::string& filename,
                  const std::vector<double>* vx = nullptr,
                  const std::vector<double>* vy = nullptr,
                  const std::vector<double>* vz = nullptr,
                  const std::string& vectorName = "heat_flux");

    /// Cubic convenience overload: n cells and spacing dx on every axis.
    bool write_3d(const std::vector<double>& scalar,
                  const std::string& scalarName,
                  std::size_t n, double dx,
                  const std::string& filename,
                  const std::vector<double>* vx = nullptr,
                  const std::vector<double>* vy = nullptr,
                  const std::vector<double>* vz = nullptr,
                  const std::string& vectorName = "heat_flux");
};

} // namespace physi_sim::io
