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
    subroutine thermal_1d_steady(n, dx, k, T_left, T_right, T_in, T_out, res_max) bind(C, name="thermal_1d_steady")
        integer(c_int), intent(in), value :: n
        real(c_double), intent(in), value :: dx, k, T_left, T_right
        real(c_double), intent(in)       :: T_in(n)
        real(c_double), intent(out)       :: T_out(n)
        real(c_double), intent(out)       :: res_max ! Max relative error

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

        ! 3. Calculate Convergence Criteria: Error_max_T / T
        res_max = 0.0
        do i = 1, n
            if (abs(T_out(i)) > 1e-12) then
                res_max = max(res_max, abs(T_out(i) - T_in(i)) / abs(T_out(i)))
            end if
        end do
    end subroutine thermal_1d_steady
    
    subroutine laplace_2d_jacobi(T, T_new, nx, ny, res_norm) bind(c, name="laplace_2d_jacobi")
        use iso_c_binding
        implicit none

        ! Interoperable arguments matching your C++ Solver2D call
        integer(c_int), value    :: nx, ny
        real(c_double), intent(in)  :: T(nx, ny)      ! Current state
        real(c_double), intent(out) :: T_new(nx, ny)  ! Next state
        real(c_double), intent(out) :: res_norm       ! L∞-norm residual

        integer :: i, j
        real(c_double) :: current_err

        res_norm = 0.0d0

        ! 1. Boundary Conditions: Preserve edges from T to T_new [cite: 5]
        T_new(1, :) = T(1, :)
        T_new(nx, :) = T(nx, :)
        T_new(:, 1) = T(:, 1)
        T_new(:, ny) = T(:, ny)

        ! 2. 5-Point Jacobi Stencil (Interior Nodes)
        ! Fortran is column-major; inner loop on 'i' maximizes cache hits
        do j = 2, ny - 1
            do i = 2, nx - 1
                ! T_new = average of 4 neighbors
                T_new(i, j) = 0.25d0 * (T(i+1, j) + T(i-1, j) + T(i, j+1) + T(i, j-1))
                
                current_err = abs(T_new(i, j) - T(i, j))   ! absolute L∞, no division
                if (current_err > res_norm) res_norm = current_err
            end do
        end do


    end subroutine laplace_2d_jacobi

    subroutine laplace_2d_tdma(T, nx, ny, res_norm) bind(c, name="laplace_2d_tdma")
    use iso_c_binding
    implicit none
    integer(c_int), value :: nx, ny
    real(c_double), intent(inout) :: T(nx, ny)
    real(c_double), intent(out) :: res_norm
    real(c_double) :: a(nx), b(nx), c(nx), d(nx), x_line(nx), T_old(nx, ny)
    integer :: i, j

    T_old = T ! Store for residual check

    ! Sweep Row-by-Row (Implicit in X, Explicit in Y)
    do j = 2, ny - 1
        do i = 1, nx
            if (i == 1 .or. i == nx) then
                b(i) = 1.0; a(i) = 0.0; c(i) = 0.0; d(i) = T(i, j) ! Dirichlet [cite: 20]
            else
                ! aW*T(i-1) + aP*T(i) + aE*T(i+1) = Source (Neighbors) [cite: 15]
                a(i) = -1.0 ! aW [cite: 16]
                c(i) = -1.0 ! aE [cite: 17]
                b(i) = 4.0  ! aP [cite: 18]
                d(i) = T(i, j-1) + T(i, j+1) ! North/South neighbors [cite: 19]
            end if
        end do
        call solve_tdma(nx, a, b, c, d, x_line) ! Reuse your core solver [cite: 6]
        T(:, j) = x_line
    end do

    ! Calculate L∞
    res_norm = 0.0
    do j = 1, ny
        do i = 1, nx
            res_norm = max(res_norm, abs(T(i,j) - T_old(i,j)))   ! absolute L∞
        end do
    end do
    end subroutine laplace_2d_tdma

end module diffusion_mod
