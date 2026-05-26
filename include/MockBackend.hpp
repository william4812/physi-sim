#pragma once
#include <iostream>
#include <atomic>
#include <vector>

/**
 * @brief Mock hardware backend for unit testing LinearDummySolver.
 *
 * Standalone — no LBM interface inheritance.
 * Audit counters allow tests to verify call counts.
 */
class MockBackend {
public:
    MockBackend() = default;

    std::atomic<int> collision_count{0};
    std::atomic<int> stream_count{0};

    void allocate(std::size_t lattice_size) {
        lattice_size_ = lattice_size;
        std::cout << "[Mock] Allocated " << lattice_size_ << " nodes.\n";
    }
    void init(size_t width, size_t height) {
        width_  = width;
        height_ = height;
        std::cout << "[Mock] Initialized " << width << "x" << height << "\n";
    }
    void collide()          { collision_count++; }
    void stream()           { stream_count++;    }
    void applyBoundaries()  {}

    void syncToHost(std::vector<double>& host_data) {
        if (host_data.empty())
            std::cout << "[Mock] Syncing zeroed-buffer to host.\n";
    }
    void compute(std::vector<double>& data, 
                 [[maybe_unused]] double alpha, 
                 [[maybe_unused]]double dt) {
        std::cout << "[MockBackend] compute called on "
                  << data.size() << " elements.\n";
    }

private:
    size_t width_{0}, height_{0}, lattice_size_{0};
};
