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
    inline double& operator()(int x, int y) noexcept {
        return data_[(y * nx_) + x];
    }

    // Fast raw access for solvers/benchmarks
    double* get_raw_data() noexcept;
    const double* get_raw_data() const noexcept;

    // 1. For initialization: Returns the actual vector object
    const std::vector<double>& get_raw_vector() const { return data_; }

    // 2. For the Fortran Bridge: Returns the raw pointer to the memory
    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }

    // 3. For the Ping-Pong Swap: Updates the internal buffer efficiently
    void update_data(const std::vector<double>& next_data) 
    {
        data_ = next_data; // This performs a vector copy
    }

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
} // namespace physi_sim
