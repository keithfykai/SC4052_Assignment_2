"""
solver.py
=========
PageRank solvers and comparison utilities.

Provides:
  - pagerank_power_iteration : sparse iterative solver
  - pagerank_closed_form     : dense direct inversion (small graphs only)
  - compare_methods          : numerical comparison between the two solutions
  - top_k_nodes              : extract top-k ranked pages
"""

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.stats import kendalltau, spearmanr


def pagerank_power_iteration(
    M: sp.csc_matrix,
    dangling: np.ndarray,
    p: float = 0.15,
    tol: float = 1e-10,
    max_iter: int = 1000,
    v: np.ndarray | None = None,
) -> tuple[np.ndarray, int, list[float]]:
    """
    Power-iteration PageRank.

    The update rule is:
        r_new = (1-p) * M * r  +  (1-p) * dangling_mass * v  +  p * v

    where dangling_mass = sum of r entries for dangling nodes, ensuring
    the operator remains stochastic.

    Parameters
    ----------
    M        : column-stochastic sparse matrix (dangling cols are zero)
    dangling : boolean mask of dangling nodes
    p        : teleport probability
    tol      : convergence threshold (L1 norm)
    max_iter : maximum iterations
    v        : personalisation / teleport vector (uniform if None)

    Returns
    -------
    r         : converged PageRank vector (sums to 1)
    n_iter    : number of iterations taken
    residuals : list of L1 residuals per iteration
    """
    n = M.shape[0]
    if v is None:
        v = np.full(n, 1.0 / n)

    r = np.full(n, 1.0 / n)
    residuals: list[float] = []

    for iteration in range(1, max_iter + 1):
        # Dangling mass redistributed uniformly (standard fix)
        dangling_mass = float(r[dangling].sum())

        r_new = (1.0 - p) * (M.dot(r) + dangling_mass * v) + p * v

        residual = float(np.abs(r_new - r).sum())
        residuals.append(residual)
        r = r_new

        if residual < tol:
            return r, iteration, residuals

    warnings.warn(f"Power iteration did not converge in {max_iter} iterations.")
    return r, max_iter, residuals


def pagerank_closed_form(
    M_dense: np.ndarray,
    p: float = 0.15,
    v: np.ndarray | None = None,
) -> np.ndarray:
    """
    Closed-form PageRank via direct matrix inversion.

    Solves:  r = p * (I - (1-p) * M)^{-1} * v

    Only practical for small graphs (n ≲ 5000).

    Parameters
    ----------
    M_dense : (n x n) column-stochastic dense numpy array
    p       : teleport probability
    v       : teleport distribution (uniform if None)

    Returns
    -------
    r : PageRank vector (normalised to sum to 1)
    """
    n = M_dense.shape[0]
    if v is None:
        v = np.full(n, 1.0 / n)

    A = np.eye(n) - (1.0 - p) * M_dense   # (I - (1-p)M)
    r = p * np.linalg.solve(A, v)          # r = p * A^{-1} * v
    r = np.abs(r) / np.abs(r).sum()        # normalise (handles tiny float errors)
    return r


def compare_methods(
    r_iter: np.ndarray, r_closed: np.ndarray
) -> dict[str, float]:
    """
    Compare power-iteration and closed-form PageRank vectors.

    Returns a dict with L1 error, L2 error, max absolute error,
    Spearman rank correlation, and Kendall tau.
    """
    l1 = float(np.abs(r_iter - r_closed).sum())
    l2 = float(np.linalg.norm(r_iter - r_closed))
    linf = float(np.abs(r_iter - r_closed).max())
    sp_corr, _ = spearmanr(r_iter, r_closed)
    kt_corr, _ = kendalltau(r_iter, r_closed)
    return {
        "L1_error": l1,
        "L2_error": l2,
        "Linf_error": linf,
        "spearman_r": float(sp_corr),
        "kendall_tau": float(kt_corr),
    }


def top_k_nodes(
    r: np.ndarray,
    idx_map: list[int],
    k: int = 10,
) -> list[tuple[int, float]]:
    """
    Return the top-k (original_node_id, pagerank_score) pairs.
    """
    top_idx = np.argsort(r)[::-1][:k]
    return [(idx_map[i], float(r[i])) for i in top_idx]
