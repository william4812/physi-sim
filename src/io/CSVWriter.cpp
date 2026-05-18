#include "io/CSVWriter.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

namespace physi_sim::io 
{

void CSVWriter::write_history(const std::string& filename, const std::vector<double>& history) 
{
    std::ofstream file(filename);

    if (!file.is_open()) 
    {
        std::cerr << "[IO Error] Failed to open " << filename << " for writing." << std::endl;
        return;
    }

    // Write header
    file << "Iteration,Residual\n";

    // Write data with scientific precision
    file << std::scientific << std::setprecision(6);
    for (size_t i = 0; i < history.size(); ++i) 
    {
        file << i << "," << history[i] << "\n";
    }

    file.close();
    std::cout << "[IO] Convergence history saved to: " << filename << std::endl;
}

// Caller owns all formatting. This method owns only file I/O + comma placement.

void CSVWriter::write_table(
        const std::string& filename,
        const std::vector<std::string>& headers,
        const std::vector<std::vector<std::string>>& rows)
{
    std::ofstream file(filename);

    if (!file.is_open())
        throw std::runtime_error("[CSVWriter] Cannot open: " + filename);

    // Header row
    for (size_t col = 0; col < headers.size(); ++col) 
    {
        file << headers[col];
        if (col + 1 < headers.size()) file << ',';
    }
    file << '\n';

    // Data rows
    for (const auto& row : rows) 
    {
        for (size_t col = 0; col < row.size(); ++col) 
        {
            file << row[col];
            if (col + 1 < row.size()) file << ',';
        }
        file << '\n';
    }

    file.close();
    std::cout << "[IO] Table saved to: " << filename
              << "  (" << rows.size() << " rows)\n";
}

} // namespace physi_sim::io
