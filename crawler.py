"""
crawler.py
==========
AI-crawler URL prioritisation tool.

Provides:
  - prioritize_crawl  : rank URLs by a composite quality-aware score,
                        respecting robots.txt crawl permissions
  - run_crawler_demo  : toy demonstration on a small URL graph
"""

import numpy as np


def prioritize_crawl(
    graph_dict: dict[str, list[str]],
    pagerank_scores: dict[str, float],
    k: int,
    allowed_map: dict[str, bool] | None = None,
    quality_features: dict[str, dict] | None = None,
    alpha: float = 0.6,
    beta: float = 0.25,
    gamma: float = 0.15,
) -> list[tuple[str, float]]:
    """
    Return the top-k URLs to crawl first, using a quality-aware heuristic.

    Scoring formula
    ---------------
        score(u) = alpha  * norm_pagerank(u)
                 + beta   * domain_trust(u)
                 + gamma  * link_density(u)

    where:
      norm_pagerank  -- PageRank score normalised to [0, 1].
      domain_trust   -- heuristic trust score (0–1) based on TLD and
                        presence in quality_features (e.g., .edu, .gov,
                        .org, or known editorial domains).
      link_density   -- out-degree normalised to [0, 1]; pages that link
                        to many others are more "hub-like" and tend to
                        be well-maintained content pages (used as a
                        lightweight quality signal in lieu of real features).

    Crawl-disallowed pages (allowed_map[url] == False) are excluded
    entirely, reflecting the robots.txt convention that disallowed pages
    must not be fetched.

    Parameters
    ----------
    graph_dict      : {url: [outlink_url, ...]}
    pagerank_scores : {url: float}
    k               : number of URLs to return
    allowed_map     : {url: bool}  — True means robots.txt allows crawling.
                      If None, all pages are treated as allowed.
    quality_features: optional dict of pre-computed features per URL
                      (e.g., {"url": {"domain_trust": 0.9, ...}}).
    alpha, beta, gamma : weighting coefficients (must sum to 1).

    Returns
    -------
    List of (url, composite_score) tuples, sorted descending.
    """
    if abs(alpha + beta + gamma - 1.0) > 1e-6:
        raise ValueError("alpha + beta + gamma must equal 1.0")

    # ── filter by robots.txt permission ─────────────────────────────────────
    candidates = list(pagerank_scores.keys())
    if allowed_map is not None:
        candidates = [u for u in candidates if allowed_map.get(u, True)]

    if not candidates:
        return []

    # ── normalise PageRank to [0, 1] ────────────────────────────────────────
    pr_vals = np.array([pagerank_scores.get(u, 0.0) for u in candidates])
    pr_range = pr_vals.max() - pr_vals.min()
    norm_pr = (pr_vals - pr_vals.min()) / pr_range if pr_range > 0 else pr_vals

    # ── domain trust heuristic ──────────────────────────────────────────────
    TRUSTED_TLDS = {".edu", ".gov", ".ac", ".org"}
    TRUSTED_DOMAINS = {
        "wikipedia.org", "arxiv.org", "nature.com", "science.org",
        "scholar.google.com", "pubmed.ncbi.nlm.nih.gov", "ieee.org",
        "acm.org", "nih.gov", "cdc.gov",
    }

    def domain_trust_score(url: str, feats: dict | None) -> float:
        if feats and "domain_trust" in feats:
            return float(feats["domain_trust"])
        url_lower = url.lower()
        for tld in TRUSTED_TLDS:
            if tld in url_lower:
                return 0.85
        for dom in TRUSTED_DOMAINS:
            if dom in url_lower:
                return 0.95
        spam_signals = ["free-", "click-", "ad-", "promo", "casino", "xxx"]
        for sig in spam_signals:
            if sig in url_lower:
                return 0.05
        return 0.5  # neutral

    trust_scores = np.array([
        domain_trust_score(u, (quality_features or {}).get(u))
        for u in candidates
    ])

    # ── link density (out-degree normalised) ────────────────────────────────
    out_degrees = np.array([
        len(graph_dict.get(u, [])) for u in candidates
    ], dtype=float)
    od_range = out_degrees.max() - out_degrees.min()
    norm_od = (out_degrees - out_degrees.min()) / od_range if od_range > 0 else out_degrees

    # ── composite score ─────────────────────────────────────────────────────
    scores = alpha * norm_pr + beta * trust_scores + gamma * norm_od
    ranked = sorted(zip(candidates, scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def run_crawler_demo() -> None:
    """Demonstrate the AI crawler prioritisation tool on a toy URL graph."""
    graph = {
        "https://arxiv.org/ml":       ["https://scholar.google.com", "https://wikipedia.org/NN"],
        "https://scholar.google.com": ["https://arxiv.org/ml"],
        "https://wikipedia.org/NN":   ["https://arxiv.org/ml", "https://spam-ads.click"],
        "https://spam-ads.click":     ["https://wikipedia.org/NN"],
        "https://govt.edu/research":  ["https://arxiv.org/ml", "https://scholar.google.com"],
        "https://casino-promo.net":   [],
    }

    pr_scores = {
        "https://arxiv.org/ml":       0.31,
        "https://scholar.google.com": 0.27,
        "https://wikipedia.org/NN":   0.19,
        "https://spam-ads.click":     0.10,
        "https://govt.edu/research":  0.09,
        "https://casino-promo.net":   0.04,
    }

    allowed_map = {
        "https://arxiv.org/ml":       True,
        "https://scholar.google.com": True,
        "https://wikipedia.org/NN":   True,
        "https://spam-ads.click":     False,   # disallowed by robots.txt
        "https://govt.edu/research":  True,
        "https://casino-promo.net":   False,   # disallowed by robots.txt
    }

    print("\n" + "=" * 60)
    print("AI CRAWLER DEMO — Top-4 URLs to crawl first")
    print("=" * 60)
    results = prioritize_crawl(graph, pr_scores, k=4, allowed_map=allowed_map)
    for rank, (url, score) in enumerate(results, 1):
        print(f"  {rank}. {url}  (composite score = {score:.4f})")
