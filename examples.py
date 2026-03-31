"""
examples.py
===========
Small hand-constructed graph examples for illustrating PageRank behaviour.

Provides:
  - example_3node_chain      : 3-node chain with shared reference
  - example_dangling_node    : 3-node graph containing a dangling node
  - example_two_communities  : two 3-node rings connected by a bridge edge
  - run_small_examples       : print PageRank results for all three examples
"""

import numpy as np
import scipy.sparse as sp

from solver import pagerank_closed_form, pagerank_power_iteration


def example_3node_chain() -> tuple[np.ndarray, list[str]]:
    """
    Example 1: 3-node chain with a shared reference node.

    Graph: A -> B, B -> C, C -> A, C -> B
    Node B receives inbound links from both A and C.

    Returns
    -------
    M      : (3 x 3) column-stochastic transition matrix
    labels : node label strings ["A", "B", "C"]
    """
    # col = source, row = destination
    # A -> B only      : col 0, row 1 = 1.0
    # B -> C only      : col 1, row 2 = 1.0
    # C -> {A, B} each : col 2, row 0 = 0.5, row 1 = 0.5
    M = np.array([
        [0.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.0],
    ])
    return M, ["A", "B", "C"]


def example_dangling_node() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Example 2: 3-node graph with a dangling (sink) node.

    Graph: A -> B, A -> C, B -> C  (C has no outlinks)

    Returns
    -------
    M        : (3 x 3) column-stochastic transition matrix
               (column 2 for C is all-zero)
    dangling : boolean mask indicating which nodes are dangling
    labels   : node label strings ["A", "B", "C"]
    """
    # col 0 (A): rows 1, 2 = 0.5 each
    # col 1 (B): row 2 = 1.0
    # col 2 (C): all zero — dangling node
    M = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 1.0, 0.0],
    ])
    dangling = np.array([False, False, True])
    return M, dangling, ["A", "B", "C"]


def example_two_communities() -> tuple[np.ndarray, list[str]]:
    """
    Example 3: Two 3-node ring communities connected by a bridge edge.

    Community 1: A -> B -> C -> A
    Community 2: D -> E -> F -> D
    Bridge:      B -> D  (B has out-degree 2: C and D)

    Returns
    -------
    M      : (6 x 6) column-stochastic transition matrix
    labels : node label strings ["A", "B", "C", "D", "E", "F"]
    """
    # Nodes: 0=A, 1=B, 2=C, 3=D, 4=E, 5=F
    M = np.zeros((6, 6))
    M[1, 0] = 1.0    # A -> B
    M[2, 1] = 0.5    # B -> C  (split with bridge)
    M[3, 1] = 0.5    # B -> D  (bridge)
    M[0, 2] = 1.0    # C -> A
    M[4, 3] = 1.0    # D -> E
    M[5, 4] = 1.0    # E -> F
    M[3, 5] = 1.0    # F -> D
    return M, ["A", "B", "C", "D", "E", "F"]


def run_small_examples() -> None:
    """Compute and print PageRank for the 3 illustrative examples."""
    p_vals = [0.05, 0.15, 0.50]

    # ── Example 1: 3-node chain ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXAMPLE 1: 3-node chain  (A->B, B->C, C->{A,B})")
    print("=" * 60)
    M, labels = example_3node_chain()
    for p in p_vals:
        r = pagerank_closed_form(M, p=p)
        print(f"  p={p:.2f}:  " +
              "  ".join(f"{lbl}={v:.4f}" for lbl, v in zip(labels, r)))

    # ── Example 2: dangling node ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXAMPLE 2: 3 nodes with dangling node C  (A->{B,C}, B->C, C=dangling)")
    print("=" * 60)
    M2, dang2, labels2 = example_dangling_node()
    M2_sp = sp.csc_matrix(M2)
    for p in p_vals:
        r_it, _, _ = pagerank_power_iteration(M2_sp, dang2, p=p)
        print(f"  p={p:.2f} (iter): " +
              "  ".join(f"{lbl}={v:.4f}" for lbl, v in zip(labels2, r_it)))

    # ── Example 3: two communities ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Two 3-node communities with bridge B->D")
    print("=" * 60)
    M3, labels3 = example_two_communities()
    for p in p_vals:
        r = pagerank_closed_form(M3, p=p)
        print(f"  p={p:.2f}:  " +
              "  ".join(f"{lbl}={v:.4f}" for lbl, v in zip(labels3, r)))
