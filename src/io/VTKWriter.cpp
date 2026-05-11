#include "io/VTKWriter.hpp"
#include <fstream>
#include <iostream>

namespace physi_sim::io 
{

void VTKWriter::write(const std::vector<double>& field,
                      const std::string& filename)
{
    std::ofstream outFile(filename);

    if (!outFile)
    {
       std::cerr << "Error: could not open file: " 
                 << filename << std::endl;
       return; 
    }

    // --- VTK Header Section ---
    outFile << "# vtk DataFile Version 3.0\n";
    outFile << "LBM-PINN Thermal Result\n";
    outFile << "ASCII\n";
    outFile << "DATASET STRUCTURED_POINTS\n";

    // --- Geometry Section ---
    outFile << "DIMENSIONS " << field.size() << " 1 1\n";
    outFile << "ORIGIN 0 0 0\n";
    outFile << "SPACING 1 0 0\n";

    // --- Attribute Section ---
    outFile << "POINT_DATA " << field.size() << "\n";
    outFile << "SCALARS Temperature double 1\n";
    outFile << "LOOKUP_TABLE default\n";

    // --- Data Section ---
    for (const auto& temp : field) 
    {
        outFile << temp << "\n";
    }

    outFile.close();
}

// Reuse the existing namespace and header structure from VTKWriter.cpp
void VTKWriter::write_2d(const double* field, int nx, int ny, const std::string& filename)
{
    std::ofstream outFile(filename);
    if (!outFile) return;

    outFile << "# vtk DataFile Version 3.0\n";
    outFile << "PhysiSim 2D Thermal Map\n";
    outFile << "ASCII\n";
    outFile << "DATASET STRUCTURED_POINTS\n";
    outFile << "DIMENSIONS " << nx << " " << ny << " 1\n";
    outFile << "ORIGIN 0 0 0\n";
    outFile << "SPACING 1.0 1.0 1.0\n"; 

    outFile << "POINT_DATA " << nx * ny << "\n";
    outFile << "SCALARS Temperature double 1\n";
    outFile << "LOOKUP_TABLE default\n";

    // Loop using the pointer and total count
    for (int i = 0; i < nx * ny; ++i) {
        outFile << field[i] << "\n";
    }
    outFile.close();
}

} // namespace physi_sim::io 
