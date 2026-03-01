#ifndef MOCK_BACKEND_HPP
#define MOCK_BACKEND_HPP

#include "IComputeBackend.hpp"
#include <iostream>
#include <atomic>

class MockBackend : public IComputeBackend 
{
  private:
    size_t width_{0}, height_{0};
    size_t lattice_size_{0};

  public:
    //MockBackend() = default;
    // Audit counters for Unit Testing
    std::atomic<int> collision_count{0};
    std::atomic<int> stream_count{0};

    // This MUST match IComputeBackend.hpp exactly
    void allocate(std::size_t lattice_size) override {
        lattice_size_ = lattice_size;
        std::cout << "[Mock] Allocated " << lattice_size_ << " nodes.\n";
    }

    void init(size_t width, size_t height) override 
    {
        width_ = width;
        height_ = height;
        std::cout << "[Mock] Initialized " << width << "x" << height << "\n";
    }

    void collide() override { collision_count++; }
    void stream() override { stream_count++; }
    void applyBoundaries() override { /* No-op */ }
    
    // Crucial for L6 Engineering: Validate data integrity
    void syncToHost(std::vector<double>& host_data) override 
    {
        if (host_data.empty()) return; 
        std::cout << "[Mock] Syncing zeroed-buffer to host for verification.\n";
    }

};

#endif //MOCK_BACKEND_HPP
