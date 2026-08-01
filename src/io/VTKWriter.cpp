#include "io/VTKWriter.hpp"
#include <fstream>
#include <iostream>

namespace physi_sim::io
{

void VTKWriter::write(const std::vector<double>& field,
                      const std::string& filename)
{
    std::ofstream outFile(filename);
    if (!outFile) { std::cerr << "Error: could not open file: " << filename << std::endl; return; }
    outFile << "# vtk DataFile Version 3.0\n" << "LBM-PINN Thermal Result\n" << "ASCII\n"
            << "DATASET STRUCTURED_POINTS\n";
    outFile << "DIMENSIONS " << field.size() << " 1 1\n" << "ORIGIN 0 0 0\n" << "SPACING 1 0 0\n";
    outFile << "POINT_DATA " << field.size() << "\n" << "SCALARS Temperature double 1\n"
            << "LOOKUP_TABLE default\n";
    for (const auto& temp : field) outFile << temp << "\n";
    outFile.close();
}

void VTKWriter::write_2d(const double* field, int nx, int ny, const std::string& filename)
{
    std::ofstream outFile(filename);
    if (!outFile) return;
    outFile << "# vtk DataFile Version 3.0\n" << "PhysiSim 2D Thermal Map\n" << "ASCII\n"
            << "DATASET STRUCTURED_POINTS\n";
    outFile << "DIMENSIONS " << nx << " " << ny << " 1\n" << "ORIGIN 0 0 0\n" << "SPACING 1.0 1.0 1.0\n";
    outFile << "POINT_DATA " << nx * ny << "\n" << "SCALARS Temperature double 1\n"
            << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nx * ny; ++i) outFile << field[i] << "\n";
    outFile.close();
}

// --- 3D XML VTImageData (.vti) cell-centred export ---------------------------
// Distinct from write()/write_2d() above (1D/2D legacy STRUCTURED_POINTS, point
// data). This is the modern XML format ParaView opens natively, carrying real
// per-axis spacing and an optional vector field. See header for the index-order
// and cell-data contracts.
bool VTKWriter::write_3d(const std::vector<double>& scalar,
                         const std::string& scalarName,
                         std::size_t nx, std::size_t ny, std::size_t nz,
                         double dx, double dy, double dz,
                         const std::string& filename,
                         const std::vector<double>* vx,
                         const std::vector<double>* vy,
                         const std::vector<double>* vz,
                         const std::string& vectorName)
{
    std::ofstream f(filename);
    if (!f) { std::cerr << "Error: could not open file: " << filename << std::endl; return false; }

    // solver flattening: (i*ny + j)*nz + k   (k, our z, fastest)
    auto id = [ny, nz](std::size_t i, std::size_t j, std::size_t l) {
        return (i * ny + j) * nz + l;
    };
    const bool hasVec = (vx && vy && vz);

    f << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
      << "  <ImageData WholeExtent=\"0 " << nx << " 0 " << ny << " 0 " << nz << "\" "
      << "Origin=\"0 0 0\" Spacing=\"" << dx << " " << dy << " " << dz << "\">\n"
      << "    <Piece Extent=\"0 " << nx << " 0 " << ny << " 0 " << nz << "\">\n"
      << "      <CellData>\n";

    f.precision(9);
    f << "        <DataArray type=\"Float64\" Name=\"" << scalarName
      << "\" format=\"ascii\">\n          ";
    for (std::size_t l = 0; l < nz; ++l)          // VTK order: x fastest, z slowest
        for (std::size_t j = 0; j < ny; ++j)
            for (std::size_t i = 0; i < nx; ++i)
                f << scalar[id(i, j, l)] << " ";
    f << "\n        </DataArray>\n";

    if (hasVec) {
        f << "        <DataArray type=\"Float64\" Name=\"" << vectorName
          << "\" NumberOfComponents=\"3\" format=\"ascii\">\n          ";
        for (std::size_t l = 0; l < nz; ++l)
            for (std::size_t j = 0; j < ny; ++j)
                for (std::size_t i = 0; i < nx; ++i) {
                    const std::size_t p = id(i, j, l);
                    f << (*vx)[p] << " " << (*vy)[p] << " " << (*vz)[p] << " ";
                }
        f << "\n        </DataArray>\n";
    }

    f << "      </CellData>\n    </Piece>\n  </ImageData>\n</VTKFile>\n";
    return static_cast<bool>(f);
}

bool VTKWriter::write_3d(const std::vector<double>& scalar,
                         const std::string& scalarName,
                         std::size_t n, double dx,
                         const std::string& filename,
                         const std::vector<double>* vx,
                         const std::vector<double>* vy,
                         const std::vector<double>* vz,
                         const std::string& vectorName)
{
    return write_3d(scalar, scalarName, n, n, n, dx, dx, dx, filename, vx, vy, vz, vectorName);
}

} // namespace physi_sim::io
