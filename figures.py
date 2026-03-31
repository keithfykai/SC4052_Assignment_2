"""
figures.py
==========
Generate and save all visualisations for the PageRank report.

Six figures are produced:
  1. convergence.png        – residual vs iteration (multiple p values, log scale)
  2. sensitivity.png        – Spearman correlation vs teleport probability p
  3. pr_distribution.png    – PageRank score distribution (log–log, power-law)
  4. top20_bar.png          – Top-20 nodes ranked by PageRank (horizontal bar)
  5. degree_dist.png        – In/out-degree distributions (log–log)
  6. degree_vs_pr.png       – Scatter: in-degree vs PageRank score (log–log)

Run with:
    python figures.py                        # uses web-Google_10k.txt by default
    python figures.py --dataset <path>       # specify a different edge-list file
    python figures.py --out-dir figs/        # save PNGs to a subdirectory
"""

import argparse
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — works everywhere
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Import project modules (must run from the project directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from graph import build_sparse_transition_matrix, load_edge_list
from solver import pagerank_power_iteration, top_k_nodes
from analysis import sensitivity_analysis


# ── Shared style ────────────────────────────────────────────────────────────
STYLE = {
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
}
plt.rcParams.update(STYLE)

P_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.85]
COLORS   = plt.cm.tab10.colors


# ── Figure 1 — Convergence ──────────────────────────────────────────────────
def fig_convergence(sa_results: dict, out: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, pv in enumerate(sorted(sa_results)):
        resids = sa_results[pv]
        ax.semilogy(range(1, len(resids) + 1), resids,
                    color=COLORS[i % len(COLORS)],
                    label=f"p = {pv}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("L₁ Residual  (log scale)")
    ax.set_title("Power-Iteration Convergence for Different Teleport Probabilities")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


# ── Figure 2 — Sensitivity ──────────────────────────────────────────────────
def fig_sensitivity(sa_results: dict, out: str) -> None:
    p_vals   = sorted(sa_results.keys())
    ref_r    = sa_results[0.15]["pagerank"]
    sp_vals  = [spearmanr(sa_results[pv]["pagerank"], ref_r)[0] for pv in p_vals]
    n_iters  = [sa_results[pv]["iterations"] for pv in p_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Spearman correlation
    ax1.plot(p_vals, sp_vals, "o-", color="steelblue", linewidth=2, markersize=7)
    ax1.axvline(0.15, color="red", linestyle="--", alpha=0.6, label="p = 0.15 (reference)")
    ax1.set_xlabel("Teleport Probability  p")
    ax1.set_ylabel("Spearman Rank Correlation with p = 0.15")
    ax1.set_title("Rank-Order Stability vs  p")
    ax1.set_ylim(0.5, 1.02)
    ax1.legend(fontsize=9)

    # Right: iterations to converge
    ax2.bar(p_vals, n_iters, color="coral", edgecolor="white", width=0.04)
    ax2.axvline(0.15, color="red", linestyle="--", alpha=0.6, label="p = 0.15")
    ax2.set_xlabel("Teleport Probability  p")
    ax2.set_ylabel("Iterations to Convergence")
    ax2.set_title("Convergence Speed vs  p")
    ax2.legend(fontsize=9)

    fig.suptitle("PageRank Sensitivity to Teleport Probability", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ── Figure 3 — PageRank score distribution (log–log) ───────────────────────
def fig_pr_distribution(r: np.ndarray, out: str) -> None:
    sorted_r = np.sort(r)[::-1]
    ranks    = np.arange(1, len(sorted_r) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ranks, sorted_r, ".", markersize=2, alpha=0.4, color="steelblue",
              label="PageRank score")

    # Fit a power-law line to the top 80 % of the rank range
    mask   = (ranks >= 5) & (ranks <= int(0.8 * len(ranks)))
    coeffs = np.polyfit(np.log10(ranks[mask]), np.log10(sorted_r[mask]), 1)
    fit_y  = 10 ** np.polyval(coeffs, np.log10(ranks))
    ax.loglog(ranks, fit_y, "r--", linewidth=1.5,
              label=f"Power-law fit  (slope ≈ {coeffs[0]:.2f})")

    ax.set_xlabel("Rank")
    ax.set_ylabel("PageRank Score")
    ax.set_title("PageRank Score Distribution  (log–log)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


# ── Figure 4 — Top-20 bar chart ─────────────────────────────────────────────
def fig_top20(r: np.ndarray, idx_map: list, out: str) -> None:
    top20 = top_k_nodes(r, idx_map, k=20)
    nodes  = [str(n) for n, _ in top20]
    scores = [s for _, s in top20]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(nodes[::-1], scores[::-1],
                   color=plt.cm.Blues(np.linspace(0.4, 0.9, 20)))
    ax.set_xlabel("PageRank Score")
    ax.set_title("Top-20 Nodes by PageRank  (web-Google 10k)")
    ax.bar_label(bars, fmt="%.2e", padding=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


# ── Figure 5 — Degree distributions ────────────────────────────────────────
def fig_degree_dist(edges: list, out: str) -> None:
    in_deg  = Counter(dst for _, dst in edges)
    out_deg = Counter(src for src, _ in edges)

    def ccdf(deg_counter):
        vals = sorted(deg_counter.values())
        n    = len(vals)
        return np.array(vals), np.arange(n, 0, -1) / n

    in_x, in_y   = ccdf(in_deg)
    out_x, out_y = ccdf(out_deg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.loglog(in_x, in_y, ".", markersize=2, alpha=0.5, color="steelblue")
    ax1.set_xlabel("In-Degree  k")
    ax1.set_ylabel("P(in-degree ≥ k)  (CCDF)")
    ax1.set_title("In-Degree CCDF  (log–log)")

    ax2.loglog(out_x, out_y, ".", markersize=2, alpha=0.5, color="coral")
    ax2.set_xlabel("Out-Degree  k")
    ax2.set_ylabel("P(out-degree ≥ k)  (CCDF)")
    ax2.set_title("Out-Degree CCDF  (log–log)")

    fig.suptitle("Degree Distributions — web-Google 10k", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ── Figure 6 — In-degree vs PageRank scatter ────────────────────────────────
def fig_degree_vs_pr(
    r: np.ndarray,
    edges: list,
    node_map: dict,
    out: str,
) -> None:
    in_deg_raw = Counter(dst for _, dst in edges)
    n = len(r)
    in_degs = np.array([in_deg_raw.get(node_map_inv, 0)
                        for node_map_inv in node_map], dtype=float)

    # node_map is original_id -> matrix_index; we need index -> original_id
    # Build array indexed by matrix position
    deg_arr = np.zeros(n)
    for orig_id, mat_idx in node_map.items():
        deg_arr[mat_idx] = in_deg_raw.get(orig_id, 0)

    # Avoid log(0)
    mask  = (deg_arr > 0) & (r > 0)
    x, y = deg_arr[mask], r[mask]

    # Bin the scatter with 2D density for readability
    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(np.log10(x), np.log10(y),
                   gridsize=40, cmap="YlOrRd", mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Node count")

    # Spearman correlation
    sp_corr, _ = spearmanr(x, y)
    ax.set_xlabel("log₁₀(In-Degree)")
    ax.set_ylabel("log₁₀(PageRank)")
    ax.set_title(f"In-Degree vs PageRank  (Spearman ρ = {sp_corr:.3f})")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PageRank figures")
    parser.add_argument("--dataset", default="web-Google_10k.txt",
                        help="Path to SNAP edge-list (.txt or .txt.gz)")
    parser.add_argument("--out-dir", default=".",
                        help="Directory to save PNG files (default: current dir)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    def out(name):
        return os.path.join(args.out_dir, name)

    print(f"\nLoading {args.dataset} …")
    edges, _, _ = load_edge_list(args.dataset)
    M, dangling, node_map, idx_map = build_sparse_transition_matrix(edges)
    n = M.shape[0]
    print(f"  {n:,} nodes,  {len(edges):,} edges")

    # Run PageRank at default p=0.15
    print("Running power iteration (p=0.15) …")
    r, n_iter, residuals = pagerank_power_iteration(M, dangling, p=0.15)
    print(f"  Converged in {n_iter} iterations")

    # Run sensitivity sweep
    print("Running sensitivity sweep …")
    sa = sensitivity_analysis(M, dangling, P_SWEEP, idx_map, top_k=10)

    # ── Generate all figures ─────────────────────────────────────────────────
    print("\nGenerating figures …")

    residuals_dict = {pv: sa[pv]["residuals"] for pv in P_SWEEP}
    fig_convergence(residuals_dict,              out("convergence.png"))
    fig_sensitivity(sa,                          out("sensitivity.png"))
    fig_pr_distribution(r,                       out("pr_distribution.png"))
    fig_top20(r, idx_map,                        out("top20_bar.png"))
    fig_degree_dist(edges,                       out("degree_dist.png"))
    fig_degree_vs_pr(r, edges, node_map,         out("degree_vs_pr.png"))

    print(f"\nAll 6 figures saved to: {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()
