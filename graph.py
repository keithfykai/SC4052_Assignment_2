"""
graph.py
========
Data loading and graph construction for the PageRank pipeline.

Provides:
  - load_edge_list     : parse SNAP-format edge lists (.txt or .txt.gz)
  - build_sparse_transition_matrix : build column-stochastic CSC matrix
"""

import gzip
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def load_edge_list(path: str) -> tuple[list[tuple[int, int]], int, int]:
    """
    Parse a SNAP-format tab-separated edge list (optionally gzipped).

    Returns
    -------
    edges  : list of (src, dst) integer tuples
    n_raw  : number of nodes as declared in the header  (0 if not found)
    e_raw  : number of edges as declared in the header  (0 if not found)
    """
    path = Path(path)
    open_fn = gzip.open if path.suffix == ".gz" else open

    edges: list[tuple[int, int]] = []
    n_raw = e_raw = 0

    with open_fn(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Try to extract "Nodes: X  Edges: Y" from comment lines
                if "Nodes:" in line and "Edges:" in line:
                    parts = line.split()
                    for i, tok in enumerate(parts):
                        if tok == "Nodes:":
                            n_raw = int(parts[i + 1])
                        if tok == "Edges:":
                            e_raw = int(parts[i + 1])
                continue
            src_str, dst_str = line.split()
            edges.append((int(src_str), int(dst_str)))

    return edges, n_raw, e_raw


def build_sparse_transition_matrix(
    edges: list[tuple[int, int]], n: int | None = None
) -> tuple[sp.csc_matrix, np.ndarray, dict[int, int], list[int]]:
    """
    Build the column-stochastic transition matrix M in CSC format.

    Dangling nodes (zero out-degree) are identified but *not* redistributed
    here; they are handled inside the power-iteration loop (standard Google
    approach).

    Returns
    -------
    M        : (n x n) column-stochastic CSC sparse matrix
               (dangling columns are all-zero — handled in power iteration)
    dangling : boolean mask, shape (n,), True where out-degree == 0
    node_map : original node-id  ->  matrix index
    idx_map  : matrix index      ->  original node-id
    """
    # Collect all node ids
    node_ids = set()
    for src, dst in edges:
        node_ids.add(src)
        node_ids.add(dst)

    if n is None:
        n = len(node_ids)

    # Map original ids to compact [0, n)
    node_map: dict[int, int] = {nid: i for i, nid in enumerate(sorted(node_ids))}
    idx_map: list[int] = [nid for nid in sorted(node_ids)]

    # Count out-degrees
    out_deg = np.zeros(n, dtype=np.float64)
    rows, cols, data = [], [], []

    for src, dst in edges:
        s = node_map[src]
        d = node_map[dst]
        rows.append(d)   # column = source, row = destination
        cols.append(s)
        data.append(1.0)
        out_deg[s] += 1.0

    # Normalise each non-dangling column
    for k in range(len(data)):
        c = cols[k]
        if out_deg[c] > 0:
            data[k] /= out_deg[c]

    M = sp.csc_matrix((data, (rows, cols)), shape=(n, n))
    dangling = out_deg == 0.0

    return M, dangling, node_map, idx_map
