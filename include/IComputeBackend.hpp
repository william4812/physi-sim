#pragma once

#include <vector>
#include <memory>

/**
 * @file IComputeBackend.hpp
 * @brief Strategy interface for LBM compute backends.
 */

/**
 * @class IComputeBackend
 * @brief Abstract base class for CPU (AVX-512) and GPU (CUDA) implementations.
 * for memory management and kernel execution (CPU vs. GPU).
 */
class IComputeBackend 
{
  public:
    // Virtual destructor is mandatory for interfaces to prevent UB during cleanup
    virtual ~IComputeBackend() = default;
    
    /**
     * @brief Allocates memory for the lattice.
     * @param lattice_size Total number of lattice nodes.
     * @note Uses [[nodiscard]] to ensure the caller handles allocation lifecycle. 
     */
    virtual void allocate(std::size_t lattice_size) = 0;
    virtual void init(size_t width, size_t height) = 0;

    /** * @name Core Physics Operators
     * These methods define the primary LBM loop phases.
     */
    ///@{
    
    /**
     * @brief Executes the Collision phase (BGK/SRT model).
     * Calculates the relaxation toward equilibrium $f_i^{eq}$. 
     * This is typically compute-bound and optimized via SIMD (CPU) or CUDA kernels (GPU).
     */
    virtual void collide() = 0;
    
    /**
     * @brief Executes the Streaming phase (Advection).
     * Handles the propagation of distributions to neighboring lattice sites.
     * This is memory-bandwidth bound; implementations must optimize for coalesced access.
     */
    virtual void stream() = 0;
    
    virtual void applyBoundaries() = 0;
    ///@}

    /**
     * @name Data Synchronization
     * Transfers lattice state between host (CPU) and acceleration device (GPU).
     */
    ///@{

    /**
     * @brief Synchronizes device memory back to the host system.
     * @param[out] host_data Target vector in system RAM. 
     * @note Principal-level optimization: Ensure vector is pre-allocated to avoid 
     * reallocations during the simulation hot-path.
     */
    virtual void syncToHost(std::vector<double>& host_data) = 0;
    ///@}
    
    /** * @name Resource Management (L6 Architectural Guards)
     * Explicitly disabled operations to enforce interface integrity.
     */
    ///@{

    /** @brief Deleted copy constructor to prevent polymorphic slicing. */
    IComputeBackend(const IComputeBackend&) = delete;

    /** @brief Deleted assignment operator to prevent illegal state duplication. */
    IComputeBackend& operator=(const IComputeBackend&) = delete;
    ///@}

  protected:
    /**
     * @brief Default constructor protected to enforce Abstract Base Class (ABC) status.
     */
    IComputeBackend() = default;
};

