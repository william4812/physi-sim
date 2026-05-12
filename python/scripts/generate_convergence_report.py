import argparse
from pathlib import Path
import sys

# Add the parent directory to the python path so we can import physi_analytics
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR.parent))

from physi_analytics import loaders, plotting

def main():
    parser = argparse.ArgumentParser(description="Generate convergence profile plots.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the C++ CSV output directory.")
    parser.add_argument("--save-dir", type=str, required=True, help="Path to save the generated figures.")
    
    args = parser.parse_args()
    
    out_path = Path(args.output_dir)
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print("[+] Loading convergence telemetry...")
        jacobi = loaders.load_convergence_csv(out_path / 'jacobi_convergence.csv')
        tdma = loaders.load_convergence_csv(out_path / 'tdma_convergence.csv')
        
        plot_file = save_path / 'convergence_comparison.png'
        plotting.plot_residuals(jacobi, tdma, plot_file)
        
    except Exception as e:
        print(f"[-] Error generating plot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
