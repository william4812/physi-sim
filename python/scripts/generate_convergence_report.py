from pathlib import Path
import plotting

# 1. Define where files live
data_dir = Path("./build")
output_dir = Path("./docs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Registry of your solvers (Update this if you add new ones)
solvers = {
    "JacobiCPU": data_dir / "JacobiCPU_100x100_convergence.csv",
    "TDMACPU": data_dir / "TDMACPU_100x100_convergence.csv",
    "JacobiGPUNoVram": data_dir / "JacobiGPUNoVram_100x100_convergence.csv",
    "JacobiGPUVram": data_dir / "JacobiGPUVram_100x100_convergence.csv",
}

# 3. Generate Individual Plots
for name, path in solvers.items():
    if path.exists():
        plotting.plot_single_convergence(path, name, output_dir / f"conv_{name}.png")

# 4. Generate Specific Pairwise Comparisons
comparisons = {
    "JacobiCPU vs TDMACPU": ["JacobiCPU", "TDMACPU"],
    "JacobiCPU vs JacobiGPUNoVram": ["JacobiCPU", "JacobiGPUNoVram"],
    "JacobiGPUNoVram vs JacobiGPUVram": ["JacobiGPUNoVram", "JacobiGPUVram"]
}

for title, keys in comparisons.items():
    # Build map for just this pair
    pair_map = {k: solvers[k] for k in keys if solvers[k].exists()}
    
    # Create filename-friendly version of title
    safe_name = title.replace(" vs ", "_vs_")
    plotting.plot_comparison_convergence(pair_map, title, output_dir / f"cmp_{safe_name}.png")
