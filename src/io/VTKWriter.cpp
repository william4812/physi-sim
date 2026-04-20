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

} // namespace physi_sim::io 
