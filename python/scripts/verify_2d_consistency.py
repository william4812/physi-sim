import argparse
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Ensure library is accessible
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from physi_analytics import loaders

def main():
    parser = argparse.ArgumentParser(description="Verify 2D consistency between Jacobi and TDMA solvers.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory containing .vtk files")
    args = parser.parse_args()
    
    out_path = Path(args.output_dir)
    jacobi_file = out_path / "jacobi_final_map.vtk"
    tdma_file = out_path / "tdma_final_map.vtk"

    try:
        # 1. Load Datasets
        j_mesh = loaders.load_thermal_vtk(jacobi_file)
        t_mesh = loaders.load_thermal_vtk(tdma_file)

        # 2. Extract Temperature Arrays
        j_temp = j_mesh.point_data["Temperature"]
        t_temp = t_mesh.point_data["Temperature"]

        # 3. Compute Analytical Difference (The "Proof")
        diff = np.abs(j_temp - t_temp)
        max_diff = np.max(diff)
        l2_norm = np.sqrt(np.sum(diff**2))

        print(f"[+] Consistency Check:")
        print(f"    - Max Absolute Difference: {max_diff:.2e}")
        print(f"    - L2 Norm of Difference:   {l2_norm:.2e}")

        # 4. Visualization
        # Reshape for 2D plotting (assumes square grid based on VTK dimensions)
        dims = j_mesh.dimensions
        temp_2d_j = j_temp.reshape(dims[0], dims[1])
        temp_2d_t = t_temp.reshape(dims[0], dims[1])
        diff_2d = diff.reshape(dims[0], dims[1])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        im1 = axes[0].imshow(temp_2d_j, cmap='jet')
        axes[0].set_title("Jacobi Solver Result")
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(temp_2d_t, cmap='jet')
        axes[1].set_title("TDMA Solver Result")
        plt.colorbar(im2, ax=axes[1])

        # Diverging map for differences
        im3 = axes[2].imshow(diff_2d, cmap='magma')
        axes[2].set_title(f"Absolute Difference\nMax: {max_diff:.2e}")
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        plt.savefig("../docs/figures/2d_consistency_check.png")
        print("[+] Comparison plot saved to ../docs/figures/2d_consistency_check.png")

    except Exception as e:
        print(f"[-] Verification failed: {e}")

if __name__ == "__main__":
    main()
