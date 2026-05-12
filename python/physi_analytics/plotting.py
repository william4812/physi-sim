import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_residuals(jacobi_df: pd.DataFrame, tdma_df: pd.DataFrame, save_path: Path):
    """Generates and saves a comparison plot with robust header detection."""
    
    def get_col(df, options):
        for opt in options:
            matches = [c for c in df.columns if c.strip().lower() == opt.lower()]
            if matches: return matches[0]
        raise KeyError(f"Could not find any of {options} in headers: {list(df.columns)}")

    try:
        # Detect columns (handles Iteration/iter and Residual/Relative_Error)
        j_iter = get_col(jacobi_df, ['Iteration', 'iter'])
        j_err = get_col(jacobi_df, ['Residual', 'Relative_Error', 'err'])
        
        t_iter = get_col(tdma_df, ['Iteration', 'iter'])
        t_err = get_col(tdma_df, ['Residual', 'Relative_Error', 'err'])

        plt.figure(figsize=(10, 6))
        plt.semilogy(jacobi_df[j_iter], jacobi_df[j_err], label='Jacobi', color='tab:blue', alpha=0.7)
        plt.semilogy(tdma_df[t_iter], tdma_df[t_err], label='LBL TDMA', color='tab:red', linewidth=2)
        
        plt.xlabel('Iteration Count')
        plt.ylabel('L-infinity Residual (log scale)')
        plt.title('Numerical Convergence Profile: Jacobi vs. TDMA')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"[+] Convergence plot saved to: {save_path}")
        plt.close()
        
    except KeyError as e:
        print(f"[-] Data Error: {e}")
        raise
