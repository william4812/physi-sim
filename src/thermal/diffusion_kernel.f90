module diffusion_mod
    use iso_c_binding
    implicit none
contains
    subroutine compute_diffusion_f90(T_now, T_next, nx, r) bind(c, name="compute_diffusion_f90")
        use iso_c_binding
        integer(c_int), value :: nx
        real(c_double), value :: r
        real(c_double), intent(in) :: T_now(nx)
        real(c_double), intent(out) :: T_next(nx)
        integer :: i
        
        print *, "DEBUG: nx received =", nx
        print *, "DEBUG: r received =", r
        print *, "DEBUG: T_now(1) =", T_now(1)

        ! The 1D Heat Equation Stencil
        do i = 2, nx-1
            T_next(i) = T_now(i) + r * (T_now(i-1) - 2.0d0*T_now(i) + T_now(i+1))
        end do

        ! Explicitly set the boundary nodes so intent(out) is fully populated
        ! Fixed Temperature (Dirichlet) Boundary Condition
        T_next(1) = T_now(1)
        T_next(nx) = T_now(nx)
    end subroutine compute_diffusion_f90

    subroutine check_abi_integrity(val_in, val_out) bind(c, name="check_abi_integrity")
        use iso_c_binding
        real(c_double), value :: val_in
        real(c_double), intent(out) :: val_out
        val_out = val_in * 2.0d0
    end subroutine check_abi_integrity

! Core TDMA Solver: Stateless and Pure for HPC/Thread-Safety
    pure subroutine solve_tdma(n, a, b, c, d, x)
        integer(c_int), intent(in) :: n
        real(c_double), intent(in) :: a(n), c(n)
        real(c_double), intent(inout) :: b(n), d(n)
        real(c_double), intent(out) :: x(n)
        
        integer :: i ! note the index i is used for both column and row, a square matrix
        real(c_double) :: m

        ! Forward Elimination (O(n) complexity)
        do i = 2, n
            m = a(i) / b(i-1)
            b(i) = b(i) - m * c(i-1) ! update term aP diagonal at i row
            d(i) = d(i) - m * d(i-1) ! update source term at i row
        end do

        ! Backward Substitution
        x(n) = d(n) / b(n) ! solve temperature at row n
        do i = n - 1, 1, -1
            x(i) = (d(i) - c(i) * x(i+1)) / b(i)
        end do
    end subroutine solve_tdma

    ! 1D Steady State Stencil Construction
    subroutine thermal_1d_steady(n, dx, k, T_left, T_right, T_out) bind(C, name="thermal_1d_steady")
        integer(c_int), intent(in), value :: n
        real(c_double), intent(in), value :: dx, k, T_left, T_right
        real(c_double), intent(out)       :: T_out(n)

        real(c_double) :: a(n), b(n), c(n), d(n)
        integer :: i

        ! Internal nodes: -aW*T(i-1) + aP*T(i) - aE*T(i+1) = 0
        do i = 2, n-1
            a(i) = -k / dx     ! West neighbor (aW)[cite: 1]
            c(i) = -k / dx     ! East neighbor (aE)[cite: 1]
            b(i) = 2.0 * k / dx ! Central (aP = aW + aE)[cite: 1]
            d(i) = 0.0          ! No source (b)[cite: 1]
        end do

        ! Boundary Conditions: Dirichlet (Fixed Temperature)
        b(1) = 1.0; c(1) = 0.0; d(1) = T_left
        b(n) = 1.0; a(n) = 0.0; d(n) = T_right

        call solve_tdma(n, a, b, c, d, T_out)
    end subroutine thermal_1d_steady

end module diffusion_mod
