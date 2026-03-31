"""
analysis.py
===========
Sensitivity analysis and visualisation for the PageRank pipeline.

Provides:
  - sensitivity_analysis : sweep over teleport probability values
  - plot_convergence     : residual-vs-iteration plot (requires matplotlib)
  - plot_sensitivity     : Spearman-correlation-vs-p plot (requires matplotlib)
"""

import time

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

from solver import pagerank_power_iteration, top_k_nodes


def sensitivity_analysis(
    M: sp.csc_matrix,
    dangling: np.ndarray,
    p_values: list[float],
    idx_map: list[int],
    top_k: int = 5,
) -> dict:
    """
    Run PageRank for each p in p_values and collect:
      - iterations to converge
      - runtime
      - top-k nodes
      - Spearman correlation with p=0.15 reference

    Returns
    -------
    results : dict keyed by p value, each entry containing
              pagerank, iterations, runtime_s, residuals,
              top_k nodes, and spearman_vs_015.
    """
    results = {}
    ref_r = None

    for p in p_values:
        t0 = time.perf_counter()
        r, n_iter, residuals = pagerank_power_iteration(M, dangling, p=p)
        elapsed = time.perf_counter() - t0

        sp_corr = None
        if ref_r is not None:
            sp_corr, _ = spearmanr(r, ref_r)
        if p == 0.15:
            ref_r = r.copy()

        results[p] = {
            "pagerank": r,
            "iterations": n_iter,
            "runtime_s": elapsed,
            "residuals": residuals,
            "top_k": top_k_nodes(r, idx_map, top_k),
            "spearman_vs_015": float(sp_corr) if sp_corr is not None else None,
        }

    return results


def plot_convergence(
    residuals_dict: dict[float, list[float]],
    save_path: str | None = None,
) -> None:
    """
    Plot residual vs iteration for multiple p values (log scale).

    Parameters
    ----------
    residuals_dict : {p_value: [residual_per_iter, ...]}
    save_path      : path to save the figure; displays interactively if None.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for pv, resids in sorted(residuals_dict.items()):
        ax.semilogy(range(1, len(resids) + 1), resids, label=f"p={pv}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L1 Residual (log scale)")
    ax.set_title("PageRank Power-Iteration Convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Convergence plot saved to {save_path}")
    else:
        plt.show()


def plot_sensitivity(
    sa_results: dict,
    save_path: str | None = None,
) -> None:
    """
    Plot Spearman rank correlation vs p (relative to p=0.15 reference).

    Parameters
    ----------
    sa_results : output of sensitivity_analysis()
    save_path  : path to save the figure; displays interactively if None.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    p_vals = sorted(sa_results.keys())
    ref_r = sa_results[0.15]["pagerank"]
    spearman_vals = []
    for pv in p_vals:
        sp_corr, _ = spearmanr(sa_results[pv]["pagerank"], ref_r)
        spearman_vals.append(sp_corr)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p_vals, spearman_vals, "o-", color="steelblue")
    ax.axvline(0.15, color="red", linestyle="--", alpha=0.5, label="p=0.15 (reference)")
    ax.set_xlabel("Teleport Probability p")
    ax.set_ylabel("Spearman Rank Correlation with p=0.15")
    ax.set_title("PageRank Sensitivity to Teleport Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Sensitivity plot saved to {save_path}")
    else:
        plt.show()
