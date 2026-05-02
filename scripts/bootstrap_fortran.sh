#!/bin/bash
# bootstrap_fortran.sh - Automates C++/Fortran Interop Plumbing

# 1. Create the Fortran Kernel with ISO_C_BINDING
cat <<EOF > src/thermal/diffusion_kernel.f90
module diffusion_mod
    use iso_c_binding
    implicit none
contains
    subroutine compute_diffusion_f90(T_now, T_next, nx, r) bind(c, name="compute_diffusion_f90")
        integer(c_int), value :: nx
        real(c_double), value :: r
        real(c_double), intent(in) :: T_now(nx)
        real(c_double), intent(out) :: T_next(nx)
        
        integer :: i
        ! The 1D Heat Equation Stencil
        do i = 2, nx-1
            T_next(i) = T_now(i) + r * (T_now(i-1) - 2.0d0*T_now(i) + T_now(i+1))
        end do
    end subroutine compute_diffusion_f90
end module diffusion_mod
EOF

# 2. Create the C++ Wrapper Header
cat <<EOF > include/thermal/FortranBackend.hpp
#pragma once
#include "IComputeBackend.hpp"

extern "C" {
    // This matches the bind(c) name in Fortran
    void compute_diffusion_f90(const double* T_now, double* T_next, int nx, double r);
}

namespace physi_sim::thermal {
class FortranBackend : public IComputeBackend {
public:
    void compute_diffusion_1d(const double* T_now, double* T_next, int nx, double r) override {
        compute_diffusion_f90(T_now, T_next, nx, r);
    }
};
}
EOF

echo "Success: Fortran kernel and C++ wrapper generated."
echo "Action: Add 'enable_language(Fortran)' to the top of your CMakeLists.txt."
