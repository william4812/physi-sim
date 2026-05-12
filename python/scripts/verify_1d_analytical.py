import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from physi_analytics import loaders

def main():
    parser = argparse.ArgumentParser(description="Verify 1D thermal results against analytical baseline.")
    parser.add_argument("--vtk-file", type=str, required=True, help="Path to the generated thermal_results.vtk.")
    parser.add_argument("--save-dir", type=str, required=True, help="Path to save the verification plot.")
    
    args = parser.parse_args()
    
    vtk_path = Path(args.vtk_file)
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"[+] Loading VTK data from: {vtk_path}")
        mesh = loaders.load_thermal_vtk(vtk_path)
        
        x = mesh.points[:, 0]
        temp = mesh.point_data["Temperature"]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, temp, label='C++ Finite Volume', linewidth=2, color='blue', marker='o', markersize=3)
        plt.plot(x, x * 1.0, 'r--', label='Analytical Baseline', alpha=0.6)
        
        plt.title("1D Thermal Diffusion Verification")
        plt.xlabel("Domain Position (x)")
        plt.ylabel("Temperature (T)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        output_plot = save_path / "verification_plot.png"
        plt.savefig(str(output_plot))
        print(f"[+] Verification plot saved to: {output_plot}")
        plt.close()
        
    except Exception as e:
        print(f"[-] Error verifying data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
