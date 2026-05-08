#pragma once
#include <vector>
#include <cstddef> // for size_t

namespace physi_sim 
{
namespace core 
{

class Grid2D 
{
public:
    /**
     * @brief Construct a new Grid 2D object
     * @param nx Number of cells in X direction (columns)
     * @param ny Number of cells in Y direction (rows)
     */
    Grid2D(int nx, int ny);

    // Standard accessors (implementation in .cpp)
    double& at(int x, int y);
    const double& at(int x, int y) const;

    // Fast raw access for solvers/benchmarks
    double* get_raw_data() noexcept;
    const double* get_raw_data() const noexcept;

    // Dimension getters - marked noexcept for compiler optimization
    int get_nx() const noexcept { return nx_; }
    int get_ny() const noexcept { return ny_; }
    size_t size() const noexcept { return data_.size(); }

private:
    int nx_;
    int ny_;
    std::vector<double> data_;

    // Flattening logic kept in header for inlining
    // Index = (Row * Width) + Column
    inline int get_index(int x, int y) const noexcept 
    {
        return (y * nx_) + x;
    }
};

} // namespace core
} // namespace physisim
