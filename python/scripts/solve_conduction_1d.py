#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import unittest
from unittest.mock import patch

class TestTDMASolver(unittest.TestCase):
    
    def test_empty_system(self):
        """Asserts n=0 logic returns the empty buffer without execution."""
        x = []
        result = solve_tdma_inplace([], [], [], [], x)
        self.assertEqual(result, [])

    def test_mismatched_dimensions(self):
        """Asserts dimensionality mismatches raise ValueError."""
        with self.assertRaises(ValueError):
            solve_tdma_inplace([1], [2, 2], [1], [3], [0, 0])

    def test_zero_diagonal_division_error(self):
        """Asserts singular matrices trigger a ZeroDivisionError."""
        a = [0, 1, 1]
        b = [0, 2, 2] # b[0] is 0
        c = [1, 1, 0]
        d = [1, 2, 3]
        x = [0, 0, 0]
        with self.assertRaises(ZeroDivisionError):
            solve_tdma_inplace(a, b, c, d, x)
            
    def test_memory_allocation_footprint(self):
        """Asserts the solver mutates arrays in-place without new allocations."""
        a, b, c, d = [0.0, -1.0], [2.0, 2.0], [-1.0, 0.0], [1.0, 1.0]
        x = [0.0, 0.0]
        
        # Capture original memory addresses
        original_c_id = id(c)
        original_d_id = id(d)
        original_x_id = id(x)
        
        solve_tdma_inplace(a, b, c, d, x)
        
        # Assert memory addresses remain identical (no new lists spawned)
        self.assertEqual(id(c), original_c_id)
        self.assertEqual(id(d), original_d_id)
        self.assertEqual(id(x), original_x_id)

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_solver_accuracy_isolated(self, mock_figure, mock_show):
        """Validates accuracy while mocking matplotlib to bypass Axes3D warnings."""
        a = [0.0, -1.0, -1.0, -1.0]
        b = [2.0, 2.0, 2.0, 2.0]
        c = [-1.0, -1.0, -1.0, 0.0]
        d = [1.0, 0.0, 0.0, 1.0]
        x = [0.0] * 4
        
        expected_x = [1.0, 1.0, 1.0, 1.0]
        solve_tdma_inplace(a, b, c, d, x)
        
        for i in range(len(x)):
            self.assertAlmostEqual(x[i], expected_x[i])
        
        # Verify the visualizations were bypassed successfully
        mock_figure.assert_not_called()
        mock_show.assert_not_called()


def getCoef(k, dx, rho, cp, dt, qDot, T0, TL, nx, theta):
    aE = np.zeros(nx)
    aW = np.zeros(nx)
    aP = np.zeros(nx)
    aP0 = np.zeros(nx)
    S = np.zeros(nx)

    # Calculate uniform interior coefficients
    diffusive_conductance = k / dx
    transient_inertia = (rho * cp * dx) / dt


    # Left Boundary (Dirichlet T = T0)
    aW[0] = 0.0
    aE[0] = diffusive_conductance
    aP0[0] = transient_inertia    
    boundary_conductance_W = k / (dx / 2.0)    
    # The known boundary temperature goes to the RHS (Source)
    S[0] = (qDot * dx) + (boundary_conductance_W * T0)
    # The central coefficient absorbs the boundary flux resistance
    aP[0] = aP0[0] + theta * (aE[0] + boundary_conductance_W)

    for i in range(1, nx - 1):
        aW[i] = diffusive_conductance
        aE[i] = diffusive_conductance
        aP0[i] = transient_inertia
        
        # Source term includes volumetric heat generation scaled by cell volume (dx)
        S[i] = qDot * dx
        
        # Central coefficient incorporates the time-stepping scheme (theta)
        aP[i] = aP0[i] + theta * (aW[i] + aE[i])

    # Right Boundary (Dirichlet T = TL)
    aW[nx - 1] = diffusive_conductance
    aE[nx - 1] = 0.0
    aP0[nx - 1] = transient_inertia    
    boundary_conductance_E = k / (dx / 2.0)
    S[nx - 1] = (qDot * dx) + (boundary_conductance_E * TL)
    aP[nx - 1] = aP0[nx - 1] + theta * (aW[nx - 1] + boundary_conductance_E)

    return aE[:], aW[:], aP[:], aP0[:], S[:]

def solve_tdma_inplace(a, b, c, d, x):
    n = len(b)
    if n == 0:
        return x
    
    if len(a) != n or len(c) != n or len(d) != n or len(x) != n:
        raise ValueError("Input arrays must have the same dimension.")
        
    if b[0] == 0:
        raise ZeroDivisionError("Zero detected on main diagonal at index 0.")

    # Forward elimination (modifies c and d in-place)
    c[0] = c[0] / b[0]
    d[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = b[i] - a[i] * c[i - 1]
        if denominator == 0.0:
            raise ZeroDivisionError(f"Zero denominator encountered during elimination at index {i}.")
        c[i] = c[i] / denominator
        d[i] = (d[i] - a[i] * d[i - 1]) / denominator

    # Back substitution (writes directly to pre-allocated x)
    x[n - 1] = d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]

    return x


def solveTDMA(a, b, c, d):
    """
    Solves a tridiagonal matrix system Ax = d using the Thomas algorithm.
    
    Parameters:
    a : list or array-like
        Lower diagonal (a[0] is ignored)
    b : list or array-like
        Main diagonal
    c : list or array-like
        Upper diagonal (c[-1] is ignored)
    d : list or array-like
        Right-hand side vector
        
    Returns:
    x : list
        Solution vector
    """
    n = len(d)

    if n == 0: return x # Handle empty input
    
    # Create copies to avoid modifying the original lists
    c_prime = [0.0] * n
    d_prime = [0.0] * n
    x = [0.0] * n

    # Normlization of th first row
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]


    # Forward elimination
    for i in range(1, n):
        # this implies a[i]*x[i-1] - a[i] * (b[i-1]/b[i-1])*x[i-1] = 0 cancelled out, so we can just use the modified coefficients
        denom = b[i] - a[i] * c_prime[i - 1] # for updated coffecient of x[i] at row i,

        if i < n - 1:
            # for updated coffecient of x[i+1] at row i, which is c[i] originally
            # because for row i - 1, c[i-1] = 0, so there is no c[i] - a[i] * c_prime[i - 1] term, so we can just use the modified coefficient
            c_prime[i] = c[i] / denom 
            
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    # Back substitution
    x[-1] = d_prime[-1] # the last element of x
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    
    return x[:] 

"""1D TRANSIENT conduction: rho*cp dT/dt = d/dx(k dT/dx) + q.
Fully implicit (backward Euler). Reuses the steady Gauss-Seidel solver as
the INNER solve at each time step; the outer loop marches in time.
"""
def run_1_D_transient_conduction_TDMA(L=0.5, k=1000.0, q_val=0.0, nx=5, TA=100, TB=500, rho=1000.0, cp=1000.0, T_initial=273.15, dt=0.1, n_steps=10000, theta=1.0):

    Tabs0 = 273.15 # K
    qDot = np.ones(nx) * q_val # W/m^3
    dx = L / nx #m
    x = np.arange(nx) * dx + 0.5 * dx#m
    T_old = np.ones(nx) * 250 + 273.15 # K
    T_new = np.ones(nx) * 250 + 273.15 # K

    # Boundary conditions
    T0 = TA + Tabs0 # K
    TL = TB + Tabs0 # K

    plt.figure("1D Transient Conduction Evolution", figsize=(10, 6))
    plot_interval = max(1, n_steps // 200)

    # 1. Precompute PDE coefficients once outside the solver loop
    # This vectorizes coefficient generation and isolates physics from linear algebra.
    aE, aW, aP, aP0, S = getCoef(k, dx, rho, cp, dt, q_val, T0, TL, nx, theta)

    # Pre-allocate TDMA vectors for the tridiagonal matrix
    a_diag = -aW.copy()
    b_diag = aP.copy()
    c_base = -aE.copy()  # We keep a pristine copy of c, as TDMA modifies it
    
    # Pre-allocate working arrays to prevent allocations inside the time loop
    c_work = np.zeros(nx)
    d_work = np.zeros(nx)

    for step in range(n_steps):
        # --- PASS 1: Direct solve using TDMA ---
        
        # Re-populate the working arrays for this specific time step
        c_work[:] = c_base[:]
        d_work[:] = (aP0 * T_old) + S

        # Execute the Thomas Algorithm in-place. 
        # T_new is passed as the solution buffer 'x'.
        solve_tdma_inplace(a_diag, b_diag, c_work, d_work, T_new)

        # Plot time snapshots with a clean alpha gradient (early = light, late = dark)
        if step % plot_interval == 0:
            alpha_val = 0.3 + 0.6 * (step / n_steps)
            # markers ONLY at cell centers = the real solved DATA
            plt.plot(x, T_new, linestyle='none', marker='o', markersize=7,
                     color='blue', alpha=alpha_val)
            # thin line = interpolation guide (no data, just visual)
            full_x = np.concatenate(([0.0], x, [L]))
            full_T = np.concatenate(([T0], T_new, [TL]))
            plt.plot(full_x, full_T, linestyle='--', color='blue',
                      alpha=alpha_val*0.5, linewidth=1)

        # update for next time step
        T_old[:] = T_new[:]

    # Final steady state verification line
    full_x = np.concatenate(([0.0], x, [L]))
    full_T = np.concatenate(([T0], T_new, [TL]))
    # markers = data, line = guide, with honest legend labels
    plt.plot(x, T_new, linestyle='none', marker='s', markersize=11, color='red',
             label='Numerical cell values (data)', zorder=5)
    plt.plot(full_x, full_T, linestyle='-', linewidth=2, color='red', alpha=0.5, 
             label='Interpolation / BC extrapolation (guide)')
    plt.plot([0.0, L], [T0, TL], linestyle='none', marker='^', markersize=12, 
             color='black', label='Dirichlet boundaries (imposed)', zorder=6)
    
    plt.xlabel('x (m)', fontsize=20)
    plt.ylabel('T (K)', fontsize=20)
    plt.title('Transient Conduction (Backward Euler, $\\theta=1.0$)', fontsize=20)
    plt.grid(True)
    plt.legend(frameon=False, loc='upper left')
    plt.tight_layout()

"""1D TRANSIENT conduction: rho*cp dT/dt = d/dx(k dT/dx) + q.
Fully implicit (backward Euler). Reuses the steady Gauss-Seidel solver as
the INNER solve at each time step; the outer loop marches in time.
"""
def run_1_D_transient_conduction(L=0.5, k=1000.0, q_val=0.0, nx=5, TA=100, TB=500, rho=1000.0, cp=1000.0, T_initial=273.15, dt=0.1, n_steps=10000, theta=1.0):

    Tabs0 = 273.15 # K
    qDot = np.ones(nx) * q_val # W/m^3
    dx = L / nx #m
    x = np.arange(nx) * dx + 0.5 * dx#m
    T_old = np.ones(nx) * 250 + 273.15 # K
    T_new = np.ones(nx) * 250 + 273.15 # K

    # Boundary conditions
    T0 = TA + Tabs0 # K
    TL = TB + Tabs0 # K

    plt.figure("1D Transient Conduction Evolution", figsize=(10, 6))
    plot_interval = max(1, n_steps // 200)

    # 1. Precompute PDE coefficients once outside the solver loop
    # This vectorizes coefficient generation and isolates physics from linear algebra.
    aE, aW, aP, aP0, S = getCoef(k, dx, rho, cp, dt, q_val, T0, TL, nx, theta)

    for step in range(n_steps):
        # --- PASS 1: update the whole field (your Gauss-Seidel sweep) ---

        # Boundary condition at x = 0
        # i = 0
        # aE = k / dx 
        # aW = 2 * k / dx
        # aP = rho * cp * dx / dt + (aW + aE) * theta
        # aP0 = rho * cp * dx / dt - (aW + aE) * (1-theta)
        # #print(f"i={i}, aE={aE}, aW={aW}, aP={aP}, qDot[i]={qDot[i]}, dx={dx}")
        # T_new[i] = (aP0 * T_old[i] + \
        #             aE * (theta*T_new[i+1] + (1-theta)*T_old[i+1]) + \
        #             aW * T0 + \
        #             qDot[i] * dx ) / aP

        # Boundary condition at x = 0
        # T0 is already absorbed into S[0] by getCoef. aW[0] evaluates to 0.0.
        T_new[0] = (aP0[0] * T_old[0] + 
                    aE[0] * (theta * T_new[1] + (1 - theta) * T_old[1]) + 
                    S[0]) / aP[0]
        
        # for i in range(1, nx - 1):
        #     aE = k / dx
        #     aW = k / dx
        #     aP = rho * cp * dx / dt + (aW + aE) * theta
        #     aP0 = rho * cp * dx / dt - (aW + aE) * (1-theta)
        #     #print(f"i={i}, aE={aE}, aW={aW}, aP={aP}, qDot[i]={qDot[i]}, dx={dx}")
        #     T_new[i] = ( aP0 * T_old[i] + \
        #                 aE * (theta*T_new[i+1] + (1-theta)*T_old[i+1]) + \
        #                 aW * (theta*T_new[i-1] + (1-theta)*T_old[i-1]) + 
        #                 qDot[i] * dx ) / aP

        # Interior nodes
        for i in range(1, nx - 1):
            T_new[i] = (aP0[i] * T_old[i] + 
                        aE[i] * (theta * T_new[i+1] + (1 - theta) * T_old[i+1]) + 
                        aW[i] * (theta * T_new[i-1] + (1 - theta) * T_old[i-1]) + 
                        S[i]) / aP[i]   
        
        # # Boundary condition at x = L
        # i = nx - 1
        # aE = 2 * k / dx
        # aW = k / dx
        # aP = rho * cp * dx / dt + (aW + aE) * theta
        # aP0 = rho * cp * dx / dt - (aW + aE) * (1-theta)
        # #print(f"i={i}, aE={aE}, aW={aW}, aP={aP}, qDot[i]={qDot[i]}, dx={dx}")
        # T_new[i] = (aP0 * T_old[i] + \
        #             aE * TL + \
        #             aW * (theta*T_new[i-1] + (1-theta)*T_old[i-1]) + 
        #             qDot[i] * dx ) / aP

        # Boundary condition at x = L
        # TL is already absorbed into S[nx-1] by getCoef. aE[nx-1] evaluates to 0.0.
        T_new[nx-1] = (aP0[nx-1] * T_old[nx-1] + 
                       aW[nx-1] * (theta * T_new[nx-2] + (1 - theta) * T_old[nx-2]) + 
                       S[nx-1]) / aP[nx-1]


        # Plot time snapshots with a clean alpha gradient (early = light, late = dark)
        if step % plot_interval == 0:
            alpha_val = 0.3 + 0.6 * (step / n_steps)
            # markers ONLY at cell centers = the real solved DATA
            plt.plot(x, T_new, linestyle='none', marker='o', markersize=7,
                     color='blue', alpha=alpha_val)
            # thin line = interpolation guide (no data, just visual)
            full_x = np.concatenate(([0.0], x, [L]))
            full_T = np.concatenate(([T0], T_new, [TL]))
            plt.plot(full_x, full_T, linestyle='--', color='blue',
                      alpha=alpha_val*0.5, linewidth=1)

        # update for next time step
        T_old[:] = T_new[:]

    # Final steady state verification line
    full_x = np.concatenate(([0.0], x, [L]))
    full_T = np.concatenate(([T0], T_new, [TL]))
    # markers = data, line = guide, with honest legend labels
    plt.plot(x, T_new, linestyle='none', marker='s', markersize=11, color='red',
             label='Numerical cell values (data)', zorder=5)
    plt.plot(full_x, full_T, linestyle='-', linewidth=2, color='red', alpha=0.5, 
             label='Interpolation / BC extrapolation (guide)')
    plt.plot([0.0, L], [T0, TL], linestyle='none', marker='^', markersize=12, 
             color='black', label='Dirichlet boundaries (imposed)', zorder=6)
    
    plt.xlabel('x (m)', fontsize=20)
    plt.ylabel('T (K)', fontsize=20)
    plt.title('Transient Conduction (Backward Euler, $\\theta=1.0$)', fontsize=20)
    plt.grid(True)
    plt.legend(frameon=False, loc='upper left')
    plt.tight_layout()


"""
This function solves the 1D conduction problem using the finite difference method.
It iteratively updates the temperature field until convergence is achieved.
"""
def run_1_D_numerical_conduction(L=0.5, k=1000.0,q_val = 0.0, nx=5, TA = 100, TB = 500):
    
    Tabs0 = 273.15 # K
    #L = 0.5 #m
    #k = 1000.0 # W/m-K
    qDot = np.ones(nx) * q_val # W/m^3
    #nx = 5
    dx = L / nx #m
    x = np.arange(nx) * dx + 0.5 * dx#m
    T = np.ones(nx) * 0 + 273.15 # K
    

    # Boundary conditions
    T0 = TA + Tabs0 # K
    TL = TB + Tabs0 # K

    residual = np.ones(nx) * 1 # K
    residualSum = np.sum(np.abs(residual))
    r_converged = 1e-9 # K
    iteration = 0

    while residualSum > r_converged:
        # --- PASS 1: update the whole field (your Gauss-Seidel sweep) ---

        # Boundary condition at x = 0
        i = 0
        aE = k / dx
        aW = 2 * k / dx
        aP = aW + aE
        #print(f"i={i}, aE={aE}, aW={aW}, aP={aP}, qDot[i]={qDot[i]}, dx={dx}")
        T[i] = ( aE * T[i+1] + aW * T0 + qDot[i] * dx ) / aP
        


        for i in range(1, nx - 1):
            aE = k / dx
            aW = k / dx
            aP = aW + aE
            #print(f"i={i}, aE={aE}, aW={aW}, aP={aP}, qDot[i]={qDot[i]}, dx={dx}")
            T[i] = ( aE * T[i+1] + aW * T[i-1] + qDot[i] * dx ) / aP
   
        # Boundary condition at x = L
        i = nx - 1
        aE = 2 * k / dx
        aW = k / dx
        aP = aW + aE
        #print(f"i={i}, aE={aE}, aW={aW}, aP={aP}, qDot[i]={qDot[i]}, dx={dx}")
        T[i] = ( aE * TL + aW * T[i-1] + qDot[i] * dx ) / aP
        

        # --- PASS 2: NOW compute residual on the complete, consistent field ---
        # Boundary condition at x = 0
        i = 0
        aE = k / dx
        aW = 2 * k / dx
        aP = aW + aE
        residual[i] = ( aE * T[i+1] + aW * T0 + qDot[i] * dx ) - aP * T[i]

        for i in range(1, nx - 1):
            aE = k / dx
            aW = k / dx
            aP = aW + aE
            residual[i] = ( aE * T[i+1] + aW * T[i-1] + qDot[i] * dx ) - aP * T[i]

        # Boundary condition at x = L
        i = nx - 1
        aE = 2 * k / dx
        aW = k / dx
        aP = aW + aE
        residual[i] = ( aE * TL + aW * T[i-1] + qDot[i] * dx ) - aP * T[i]

        residualSum = np.sum(np.abs(residual)) #np.abs(T-T_old)/max(np.abs(T))
        iteration += 1
        print("Residual = {0:11.9f} at Iteration {1:d}".format(residualSum, iteration))

        if (residualSum <= r_converged):
            print("Converged after {0:d} iterations".format(iteration))
            break

    front_T = [T0]
    back_T = [TL]
    resultT = np.concatenate((front_T, T, back_T))
    fron_x = [0]
    back_x = [L]
    resultX = np.concatenate((fron_x, x, back_x))

    # Return the required arrays and length for plotting
    return resultX, resultT, nx


def run_1_D_analytical_conduction(L=0.5, k=1000.0,q_val = 0.0, nx = 50, TA = 100, TB = 500):
    Tabs0 = 273.15 # K
    #L = 0.5 #m
    #k = 1000.0 # W/m-K
    qDot = q_val # W/m^3
    #nx = 50
    dx = L / nx #m  
    x = np.arange(nx) * dx + 0.5 * dx#m
    T = np.ones(nx) * 0 + 273.15 # K

    # Boundary conditions
    T0 = TA + Tabs0 # K
    TL = TB + Tabs0 # K

    for i in range(nx):
        T[i] = q_val/(2*k)*(L - x[i])*x[i] + (TL - T0 )/L * x[i] + T0
        #T[i] = T0 + (TL - T0) * (x[i] / L)

    front_T = [T0]
    back_T = [TL]
    resultT = np.concatenate((front_T, T, back_T))
    fron_x = [0]
    back_x = [L]
    resultX = np.concatenate((fron_x, x, back_x))

    # Return the required arrays and length for plotting
    return resultX, resultT, nx

def plot_results(resultX, resultT, nx):
    # 1. Set global publication-quality styles
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.rcParams['font.weight'] = 'bold'          # Bold text globally
    plt.rcParams['axes.labelweight'] = 'bold'     # Bold labels
    plt.rcParams['axes.titleweight'] = 'bold'     # Bold titles

    #plt.plot(x, T, marker='o', color='b', label='T(x)')
    plt.plot(resultX, resultT, linestyle='--', linewidth=4, 
            marker='o', markersize=10, color='b', label='T(x) - Numerical')
    #plt.plot(0, T0, marker='o', markersize = 16, color='r', label='T0 = 100°C')
    #plt.plot(L, TL, marker='o', markersize = 16, color='r', label='TL = 500°C')
    plt.xlim(0, resultX[nx+1]*1.1)
    plt.ylim(0, resultT[nx+1]*1.1)
    plt.xlabel('x (m)', fontsize=20)
    plt.ylabel('T (K)', fontsize=20)
    plt.legend(fontsize=20, frameon=False, loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_1_D_conduction_no_heat_generation():
     # Unpack the returned variables from the analytical function
    resultX_ana, resultT_ana, nx = run_1_D_analytical_conduction(L=0.5, k=1000.0,q_val = 0.0, nx = 50, TA = 100, TB = 500)
    # Unpack the returned variables from the simulation function
    resultX, resultT, nx = run_1_D_numerical_conduction(L=0.5, k=1000.0,q_val = 0.0, nx = 5, TA = 100, TB = 500)    

    plt.figure("1_D Conduction: No Heat Generation", figsize=(10, 16))
    # 1. Set global publication-quality styles
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.rcParams['font.weight'] = 'bold'          # Bold text globally
    plt.rcParams['axes.labelweight'] = 'bold'     # Bold labels
    plt.rcParams['axes.titleweight'] = 'bold'     # Bold titles

    plt.plot(resultX_ana, resultT_ana, linestyle='-', linewidth=3, 
             color='r', label='T(x) - Analytical')
    plt.plot(resultX, resultT, linestyle='--', linewidth=6, 
            marker='o', markersize=10, color='k', label='T(x) - Numerical')

    plt.xlim(0, np.max(resultX)*1.1)
    plt.ylim(0, np.max(resultT)*1.1)
    plt.xlabel('x (m)', fontsize=20)
    plt.ylabel('T (K)', fontsize=20)
    plt.legend(fontsize=20, frameon=False, loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    #plt.show()

def run_1_D_conduction_uniform_heat_generation():
     # Unpack the returned variables from the analytical function
    resultX_ana, resultT_ana, nx = run_1_D_analytical_conduction(L=0.02, k=0.5,q_val = 1000.0e3, nx = 50, TA = 100, TB = 200)
    # Unpack the returned variables from the simulation function
    resultX, resultT, nx = run_1_D_numerical_conduction(L=0.02, k=0.5,q_val = 1000.0e3, nx = 5, TA = 100, TB = 200)    

    plt.figure("1_D Conduction: Uniform Heat Generation", figsize=(10, 16))
    # 1. Set global publication-quality styles
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.rcParams['font.weight'] = 'bold'          # Bold text globally
    plt.rcParams['axes.labelweight'] = 'bold'     # Bold labels
    plt.rcParams['axes.titleweight'] = 'bold'     # Bold titles

    plt.plot(resultX_ana, resultT_ana, linestyle='-', linewidth=3, 
             color='r', label='T(x) - Analytical')
    plt.plot(resultX, resultT, linestyle='--', linewidth=6, 
            marker='o', markersize=10, color='k', label='T(x) - Numerical')

    plt.xlim(0, np.max(resultX)*1.1)
    plt.ylim(0, np.max(resultT)*1.1)
    plt.xlabel('x (m)', fontsize=20)
    plt.ylabel('T (K)', fontsize=20)
    plt.legend(fontsize=20, frameon=False, loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    #plt.show()

def test_solveTDMA():
    """Run this BEFORE wiring solveTDMA into the transient loop.
    Expected x for every case was cross-checked against np.linalg.solve."""
    cases = [
        ("A: n=1 trivial",    [0],       [3],       [0],       [9],         [3]),
        ("B: n=2 minimal",    [0,1],     [2,3],     [1,0],     [4,7],       [1,2]),
        ("C: n=3 symmetric",  [0,1,1],   [4,4,4],   [1,1,0],   [5,6,5],     [1,1,1]),
        ("D: n=4 asymmetric", [0,2,1,3], [5,6,7,4], [1,3,2,0], [9,7,21,11], [2,-1,3,0.5]),
    ]
    all_passed = True
    for name, a, b, c, d, expected in cases:
        got = solveTDMA(np.array(a, float), np.array(b, float),
                         np.array(c, float), np.array(d, float))
        ok = np.allclose(got, expected, atol=1e-9)
        all_passed &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {np.round(got,6)}, expected {expected}")
    print("\nALL PASSED" if all_passed else "\nSOME FAILED — do not wire into the time loop yet")



# This is a standard function you define yourself
def main():

    #run_1_D_conduction_no_heat_generation()

    #run_1_D_conduction_uniform_heat_generation()

    #run_1_D_transient_conduction()

    run_1_D_transient_conduction_TDMA()

    plt.show()  # Show all plots at once



# This guard checks if the script is being executed directly
if __name__ == "__main__":
    main()
        
    #test_solveTDMA()

    #suite = unittest.TestLoader().loadTestsFromTestCase(TestTDMASolver)
    #unittest.TextTestRunner(verbosity=2).run(suite)
