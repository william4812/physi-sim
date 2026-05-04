import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# 1. Robust Path Discovery
# This finds the project root by looking 2 levels up from this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
VTK_PATH = PROJECT_ROOT / "build" / "thermal_results.vtk"

def main():
    if not VTK_PATH.exists():
        print(f"[-] Error: VTK file not found at {VTK_PATH}")
        print("    Ensure you have run the C++ solver and generated the result.")
        sys.exit(1)

    # 2. Read the VTK file
    print(f"[+] Loading data from: {VTK_PATH}")
    mesh = pv.read(str(VTK_PATH))
    
    # Extract coordinates and Temperature
    # Point data key must match the string in your C++ VTKWriter
    x = mesh.points[:, 0]
    temp = mesh.point_data["Temperature"]

    # 3. Plotting (The MATLAB-style way)
    plt.figure(figsize=(10, 6))
    plt.plot(x, temp, label='C++ Finite Volume', linewidth=2, color='blue', marker='o', markersize=3)
    
    # Analytical Reference (Linear ramp for initial verification)
    # This works as long as 'x' is a numpy array
    analytical = x * 1.0 
    plt.plot(x, analytical, 'r--', label='Analytical Baseline', alpha=0.6)

    plt.title("1D Thermal Diffusion Verification")
    plt.xlabel("Domain Position (x)")
    plt.ylabel("Temperature (T)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the result in the build directory alongside the data
    output_plot = PROJECT_ROOT / "build" / "verification_plot.png"
    plt.savefig(str(output_plot))
    print(f"[+] Plot saved to: {output_plot}")
    plt.show()

if __name__ == "__main__":
    main()
