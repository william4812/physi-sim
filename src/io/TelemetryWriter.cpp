#include "io/TelemetryWriter.hpp"
#include <fstream>
#include <iostream>

namespace physi_sim::io {

TelemetryWriter::TelemetryWriter(std::string output_dir) 
    : output_path_(std::move(output_dir)) {
    ensure_directory();
}

void TelemetryWriter::ensure_directory() const {
    if (!std::filesystem::exists(output_path_)) {
        std::filesystem::create_directories(output_path_);
    }
}

bool TelemetryWriter::save_convergence(const std::vector<double>& history, 
                                      const std::string& filename) const {
    std::filesystem::path full_path = output_path_ / filename;
    std::ofstream file(full_path);

    if (!file.is_open()) {
        return false;
    }

    file << "Iteration,Relative_Error\n";
    for (size_t i = 0; i < history.size(); ++i) {
        file << (i + 1) << "," << history[i] << "\n";
    }

    std::cout << "[IO] Telemetry saved to: " << full_path << "\n";
    return true;
}

} // namespace physi_sim::io
