#pragma once
#include <vector>
#include <string>
#include <filesystem>

namespace physi_sim::io 
{

/**
 * @class TelemetryWriter
 * @brief Handles architectural separation of simulation metadata and I/O.
 */
class TelemetryWriter {
public:
    explicit TelemetryWriter(std::string output_dir = "output");

    /**
     * @brief Saves convergence residuals to a CSV for post-processing.
     * @param history Vector of relative residuals per iteration.
     * @param filename Base filename (defaults to convergence.csv).
     */
    bool save_convergence(const std::vector<double>& history, 
                          const std::string& filename = "convergence.csv") const;

private:
    std::filesystem::path output_path_;
    void ensure_directory() const;
};

} // namespace physi_sim::io
