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
# ── ADD TO python/physi_analytics/plotting.py ─────────────────────────────────
# Two new functions — append after plot_residuals(). Nothing else changes.

import numpy as np

def plot_benchmark_bars(profiling_df: pd.DataFrame, save_path: Path):
    """
    Bar chart: iterations and wall time per solver.
    Reads ProfilingHarness::writeCSV() output.

    Columns expected: solver, iterations, wall_time_ms, converged
    """
    df = profiling_df[profiling_df["converged"] == True].copy()
    names  = df["solver"].tolist()
    iters  = df["iterations"].tolist()
    times  = df["wall_time_ms"].tolist()

    colors = {
        "JacobiCPU": "#4C72B0",
        "TDMACPU":   "#55A868",
        "JacobiGPU": "#DD8452",
    }
    bar_colors = [colors.get(n, "#888888") for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CPU vs GPU Solver Benchmark — 100×100 grid, normalized L∞ < 1×10⁻⁴",
                 fontweight="bold")

    # Panel 1: iteration counts
    bars1 = ax1.bar(names, iters, color=bar_colors, alpha=0.85, edgecolor="white")
    ax1.set_ylabel("Iterations to convergence")
    ax1.set_title("Iteration Count")
    ax1.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars1, iters):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 20,
                 f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 2: wall time
    bars2 = ax2.bar(names, times, color=bar_colors, alpha=0.85, edgecolor="white")
    ax2.set_ylabel("Wall time (ms)")
    ax2.set_title("Wall Time")
    ax2.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars2, times):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                 f"{v:.1f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Annotate GPU speedup if JacobiCPU and JacobiGPU both present
    name_time = dict(zip(names, times))
    if "JacobiCPU" in name_time and "JacobiGPU" in name_time:
        speedup = name_time["JacobiCPU"] / name_time["JacobiGPU"]
        ax2.annotate(f"GPU {speedup:.1f}× faster",
                     xy=(names.index("JacobiGPU"), name_time["JacobiGPU"]),
                     xytext=(len(names)/2, max(times)*0.75),
                     arrowprops=dict(arrowstyle="->", color="#DD8452"),
                     fontsize=10, color="#DD8452", fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    print(f"[+] Benchmark chart saved to: {save_path}")
    plt.close(fig)


def plot_convergence_with_gpu(jacobi_df: pd.DataFrame,
                               tdma_df: pd.DataFrame,
                               profiling_df: pd.DataFrame,
                               save_path: Path):
    """
    Extended convergence plot: CPU curves + GPU final point annotated.
    Re-uses plot_residuals logic and adds GPU annotation from ProfilingHarness CSV.
    """
    def get_col(df, options):
        for opt in options:
            matches = [c for c in df.columns if c.strip().lower() == opt.lower()]
            if matches: return matches[0]
        raise KeyError(f"Could not find any of {options} in {list(df.columns)}")

    j_iter = get_col(jacobi_df, ["Iteration", "iter"])
    j_err  = get_col(jacobi_df, ["Residual", "Relative_Error", "err"])
    t_iter = get_col(tdma_df,   ["Iteration", "iter"])
    t_err  = get_col(tdma_df,   ["Residual", "Relative_Error", "err"])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.semilogy(jacobi_df[j_iter], jacobi_df[j_err],
                label=f"JacobiCPU  ({len(jacobi_df):,} iters)",
                color="#4C72B0", alpha=0.8, linewidth=1.5)
    ax.semilogy(tdma_df[t_iter], tdma_df[t_err],
                label=f"TDMACPU  ({len(tdma_df):,} iters)",
                color="#55A868", linewidth=2)

    # Annotate GPU final residual as a point (no per-iteration history yet)
    if profiling_df is not None:
        gpu_row = profiling_df[profiling_df["solver"] == "JacobiGPU"]
        if not gpu_row.empty:
            gpu_iters = int(gpu_row["iterations"].iloc[0])
            gpu_res   = float(gpu_row["final_residual"].iloc[0])
            ax.scatter([gpu_iters], [gpu_res], color="#DD8452", s=80, zorder=5,
                       label=f"JacobiGPU  ({gpu_iters:,} iters, final point)")
            ax.annotate(f"GPU converged\n{gpu_iters:,} iters",
                        xy=(gpu_iters, gpu_res),
                        xytext=(gpu_iters * 0.5, gpu_res * 50),
                        arrowprops=dict(arrowstyle="->", color="#DD8452"),
                        fontsize=8, color="#DD8452")

    ax.set_xlabel("Iteration Count")
    ax.set_ylabel("L∞ Residual (absolute, log scale)")
    ax.set_title("Convergence Consistency: JacobiCPU · TDMACPU · JacobiGPU\n"
                 "Same 50×50 grid · same boundary conditions · absolute L∞",
                 fontweight="bold")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    print(f"[+] Extended convergence plot saved to: {save_path}")
    plt.close(fig)
