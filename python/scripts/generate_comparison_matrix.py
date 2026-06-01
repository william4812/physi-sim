import sys
from pathlib import Path
import itertools

# Standard path setup
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR.parent))

from physi_analytics import loaders, plotting

def main():
    # Map friendly names to the actual file prefixes present in your build/ folder
    # Note: These must match the filenames seen in your 'ls' output
    file_map = {
        "JacobiCPU": "cmp_cpu_jacobi_100",
        "TDMACPU": "cmp_cpu_tdma_100",
        "JacobiGPUNoVram": "cmp_gpu_jacobi_novram_100",
        "JacobiGPUVram": "cmp_gpu_jacobi_vram_100"
    }
    
    # Base directory where .vtk files are located
    data_dir = Path(".") 
    save_dir = Path("../docs/figures/comparisons")
    save_dir.mkdir(parents=True, exist_ok=True)

    solvers = list(file_map.keys())
    
    # Iterate through combinations
    for s1, s2 in itertools.combinations(solvers, 2):
        file1 = data_dir / f"{file_map[s1]}.vtk"
        file2 = data_dir / f"{file_map[s2]}.vtk"
        
        if file1.exists() and file2.exists():
            print(f"Generating comparison: {s1} vs {s2}")
            save_path = save_dir / f"cmp_{s1}_vs_{s2}.png"
            
            # Using your existing plotting library
            plotting.plot_generic_comparison(
                loaders.load_thermal_vtk(file1),
                loaders.load_thermal_vtk(file2),
                s1, s2,
                save_path
            )
        else:
            print(f"Skipping {s1} vs {s2}: Missing .vtk files")

if __name__ == "__main__":
    main()
