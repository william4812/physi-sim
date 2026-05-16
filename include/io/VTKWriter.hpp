#pragma once
#include <vector>
#include <string>

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
};

} // namespace physi_sim::io
