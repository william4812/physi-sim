// TODO: migrate to TelemetryWriter — CSVWriter is redundant (tracked in cleanup)
#pragma once
#include <vector>
#include <string>

namespace physi_sim::io 
{

class CSVWriter 
{
public:
    /**
     * @brief Writes residual history to a CSV file.
     * @param filename Path to the output file.
     * @param history Vector of residuals per iteration.
     */
    void write_history(const std::string& filename, 
                       const std::vector<double>& history);
    // generic table writer.
    // headers : column names for the first row
    // rows    : each inner vector is one data row of pre-formatted strings
    //
    // All numeric formatting is the caller's responsibility.
    // CSVWriter owns only: open file, write commas, close file.
    // Throws std::runtime_error if the file cannot be opened.
    static void write_table(
            const std::string& filename,
            const std::vector<std::string>& headers,
            const std::vector<std::vector<std::string>>& rows);
};

} // namespace physi_sim:io
