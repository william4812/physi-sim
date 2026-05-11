#pragma once
#include <vector>
#include <string>

namespace physi_sim 
{
namespace io 
{

class CSVWriter 
{
public:
    /**
     * @brief Writes residual history to a CSV file.
     * @param filename Path to the output file.
     * @param history Vector of residuals per iteration.
     */
    void write_history(const std::string& filename, const std::vector<double>& history);
};

} // namespace io
} // namespace physi_sim
