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

    // Standard accessors
    double& at(int x, int y);
    const double& at(int x, int y) const;
    
    inline double& operator()(int x, int y) noexcept {
        return data_[get_index(x, y)];
    }
    
    inline const double& operator()(int x, int y) const noexcept {
        return data_[get_index(x, y)];
    }

    // Fast raw access for solvers/benchmarks
    double* get_raw_data() noexcept;
    const double* get_raw_data() const noexcept;

    const std::vector<double>& get_raw_vector() const { return data_; }

    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }

    void update_data(const std::vector<double>& next_data) 
    {
        data_ = next_data; 
    }

    int get_nx() const noexcept { return nx_; }
    int get_ny() const noexcept { return ny_; }
    size_t size() const noexcept { return data_.size(); }

private:
    int nx_;
    int ny_;
    std::vector<double> data_;

    // Return std::size_t directly to match vector indexing and avoid sign-conversion warnings
    inline std::size_t get_index(int x, int y) const noexcept 
    {
        return (static_cast<std::size_t>(y) * static_cast<std::size_t>(nx_)) + static_cast<std::size_t>(x);
    }
};

} // namespace core
} // namespace physi_sim