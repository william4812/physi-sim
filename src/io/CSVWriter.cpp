#include "io/CSVWriter.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

namespace physi_sim 
{
namespace io 
{

void CSVWriter::write_history(const std::string& filename, const std::vector<double>& history) 
{
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "[IO Error] Failed to open " << filename << " for writing." << std::endl;
        return;
    }

    // Write header
    file << "Iteration,Residual\n";

    // Write data with scientific precision
    file << std::scientific << std::setprecision(6);
    for (size_t i = 0; i < history.size(); ++i) {
        file << i << "," << history[i] << "\n";
    }

    file.close();
    std::cout << "[IO] Convergence history saved to: " << filename << std::endl;
}

} // namespace io
} // namespace physi_sim
