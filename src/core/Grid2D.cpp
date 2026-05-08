#include "core/Grid2D.hpp"
#include <stdexcept>
#include <cassert>

namespace physi_sim 
{
namespace core 
{

Grid2D::Grid2D(int nx, int ny) 
    : nx_(nx), ny_(ny), data_(static_cast<size_t>(nx) * ny, 0.0) 
{
    if (nx <= 0 || ny <= 0) 
    {
        throw std::invalid_argument("Grid dimensions must be positive.");
    }
}

double& Grid2D::at(int x, int y) 
{
    // Debug-only bounds checking for performance
    assert(x >= 0 && x < nx_ && y >= 0 && y < ny_);
    return data_[get_index(x, y)];
}

const double& Grid2D::at(int x, int y) const 
{
    assert(x >= 0 && x < nx_ && y >= 0 && y < ny_);
    return data_[get_index(x, y)];
}

double* Grid2D::get_raw_data() noexcept 
{
    return data_.data();
}

const double* Grid2D::get_raw_data() const noexcept 
{
    return data_.data();
}

} // namespace core
} // namespace physisim
