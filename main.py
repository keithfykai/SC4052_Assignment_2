"""
main.py
=======
CLI entry point and full pipeline orchestration for the PageRank project.

Usage
-----
  # Run small illustrative examples
  python main.py --examples

  # Run on the 10k dataset with default settings
  python main.py --dataset web-Google_10k.txt --p 0.15 --top 10

  # Also compute closed-form solution (small graphs only)
  python main.py --dataset web-Google_10k.txt --closed-form --no-sensitivity

  # Run on the full web-Google dataset (~875k nodes)
  python main.py --dataset web-Google.txt.gz --p 0.15 --top 10

  # Run the AI crawler prioritisation demo
  python main.py --crawler-demo

  # Generate and save convergence / sensitivity plots (requires matplotlib)
  python main.py --dataset web-Google_10k.txt --plots
"""

import argparse
import time

from scipy.stats import spearmanr

from analysis import plot_convergence, plot_sensitivity, sensitivity_analysis
from crawler import run_crawler_demo
from examples import run_small_examples
from graph import build_sparse_transition_matrix, load_edge_list
from solver import (
    compare_methods,
    pagerank_closed_form,
    pagerank_power_iteration,
    top_k_nodes,
)


def run_pipeline(
    dataset_path: str,
    p: float = 0.15,
    top_k: int = 10,
    run_closed_form: bool = False,
    sensitivity: bool = True,
):
    """
    Full pipeline: load -> build matrix -> power-iterate -> report results.

    Parameters
    ----------
    dataset_path    : path to SNAP edge-list file (.txt or .txt.gz)
    p               : teleport probability
    top_k           : number of top nodes to display
    run_closed_form : also solve via direct inversion (only for n ≤ 5000)
    sensitivity     : run sensitivity sweep across multiple p values

    Returns
    -------
    r        : converged PageRank vector
    n_iter   : iterations taken
    residuals: per-iteration L1 residuals
    idx_map  : mapping from matrix index to original node id
    """
    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_path}")
    print(f"Teleport: p = {p}")
    print(f"{'='*60}")

    # ── Load ─────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    edges, n_hdr, e_hdr = load_edge_list(dataset_path)
    t_load = time.perf_counter() - t0
    print(f"Loaded {len(edges):,} edges in {t_load:.2f}s  "
          f"(header says {n_hdr:,} nodes, {e_hdr:,} edges)")

    # ── Build matrix ─────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    M, dangling, node_map, idx_map = build_sparse_transition_matrix(edges)
    n = M.shape[0]
    t_build = time.perf_counter() - t1
    print(f"Matrix built: {n:,} x {n:,},  "
          f"{M.nnz:,} non-zeros,  "
          f"{dangling.sum():,} dangling nodes  ({t_build:.2f}s)")

    # ── Power iteration ───────────────────────────────────────────────────────
    t2 = time.perf_counter()
    r, n_iter, residuals = pagerank_power_iteration(M, dangling, p=p)
    t_iter = time.perf_counter() - t2
    print(f"Power iteration: {n_iter} iterations  ({t_iter:.3f}s)  "
          f"final residual = {residuals[-1]:.2e}")

    # ── Top-k ─────────────────────────────────────────────────────────────────
    top = top_k_nodes(r, idx_map, top_k)
    print(f"\nTop-{top_k} nodes (p={p}):")
    print(f"  {'Rank':>4}  {'Node':>10}  {'PageRank':>12}")
    for rank, (node, score) in enumerate(top, 1):
        print(f"  {rank:>4}  {node:>10}  {score:>12.8f}")

    # ── Closed-form (small graphs only) ──────────────────────────────────────
    if run_closed_form and n <= 5000:
        print("\nComputing closed-form solution ...")
        M_dense = M.toarray()
        t3 = time.perf_counter()
        r_cf = pagerank_closed_form(M_dense, p=p)
        t_cf = time.perf_counter() - t3
        metrics = compare_methods(r, r_cf)
        print(f"  Closed-form time: {t_cf:.3f}s")
        print(f"  L1 error  : {metrics['L1_error']:.2e}")
        print(f"  L2 error  : {metrics['L2_error']:.2e}")
        print(f"  Linf error: {metrics['Linf_error']:.2e}")
        print(f"  Spearman r: {metrics['spearman_r']:.8f}")
        print(f"  Kendall τ : {metrics['kendall_tau']:.8f}")

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    if sensitivity:
        p_vals = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.85]
        print("\nSensitivity analysis across p values:")
        print(f"  {'p':>6}  {'Iters':>6}  {'Time(s)':>8}  {'Spearman vs p=0.15':>20}")
        sa = sensitivity_analysis(M, dangling, p_vals, idx_map, top_k)
        ref_r = sa[0.15]["pagerank"]
        for pv in p_vals:
            res = sa[pv]
            sp_corr, _ = spearmanr(res["pagerank"], ref_r)
            print(f"  {pv:>6.2f}  {res['iterations']:>6}  "
                  f"{res['runtime_s']:>8.3f}  {sp_corr:>20.6f}")

    return r, n_iter, residuals, idx_map


def main() -> None:
    parser = argparse.ArgumentParser(description="PageRank — SC4052 Assignment 2")
    parser.add_argument("--dataset", type=str, default="web-Google_10k.txt",
                        help="Path to edge-list file (.txt or .txt.gz)")
    parser.add_argument("--p", type=float, default=0.15,
                        help="Teleport probability (default 0.15)")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top nodes to report")
    parser.add_argument("--closed-form", action="store_true",
                        help="Also compute closed-form (graphs with n ≤ 5000 only)")
    parser.add_argument("--no-sensitivity", action="store_true",
                        help="Skip sensitivity analysis")
    parser.add_argument("--examples", action="store_true",
                        help="Run small illustrative examples")
    parser.add_argument("--crawler-demo", action="store_true",
                        help="Run AI crawler prioritisation demo")
    parser.add_argument("--plots", action="store_true",
                        help="Generate and save convergence/sensitivity plots")
    args = parser.parse_args()

    if args.examples:
        run_small_examples()

    if args.crawler_demo:
        run_crawler_demo()

    r, n_iter, residuals, idx_map = run_pipeline(
        dataset_path=args.dataset,
        p=args.p,
        top_k=args.top,
        run_closed_form=args.closed_form,
        sensitivity=not args.no_sensitivity,
    )

    if args.plots:
        edges, _, _ = load_edge_list(args.dataset)
        M, dangling, _, idx_map2 = build_sparse_transition_matrix(edges)
        p_vals = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.85]
        sa = sensitivity_analysis(M, dangling, p_vals, idx_map2, top_k=10)
        residuals_dict = {pv: sa[pv]["residuals"] for pv in p_vals}
        plot_convergence(residuals_dict, save_path="convergence.png")
        plot_sensitivity(sa, save_path="sensitivity.png")


if __name__ == "__main__":
    main()
