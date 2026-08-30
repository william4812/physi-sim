#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


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


# This is a standard function you define yourself
def main():

    run_1_D_conduction_no_heat_generation()

    run_1_D_conduction_uniform_heat_generation()

    plt.show()  # Show all plots at once



# This guard checks if the script is being executed directly
if __name__ == "__main__":
    main()