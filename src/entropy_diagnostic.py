"""
Attention-Entropy Diagnostic (Lock Detection)

Detects when a retriever is "locked" — attending to a narrow, repetitive
set of nodes instead of adapting to the query. This is the engineering
implementation of the "attention lock" concept from existential systems theory.

Key insight: A healthy retriever should have query-dependent attention
distribution. If the entropy of retrieved scores is consistently low
across diverse queries, the retriever is locked.

Metrics:
- H(scores): Shannon entropy of the score distribution (softmax-normalized)
- Lock ratio: fraction of queries where H < threshold
- Diversity@k: number of unique content clusters in top-k
- Attention stability: std of H across queries (low = locked pattern)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.special import softmax


def score_entropy(scores: np.ndarray, temperature: float = 1.0) -> float:
    """
    Compute Shannon entropy of the retrieval score distribution.
    
    Higher entropy = more diverse attention (healthy).
    Lower entropy = concentrated on few nodes (potentially locked).
    
    Args:
        scores: Raw retrieval scores for all nodes.
        temperature: Softmax temperature. Lower = sharper distribution.
    
    Returns:
        Shannon entropy in nats.
    """
    # Softmax normalization to get probability distribution
    probs = softmax(scores / temperature)
    # Remove zeros for log stability
    probs = probs[probs > 1e-10]
    return -np.sum(probs * np.log(probs))


def max_entropy(n: int) -> float:
    """Maximum possible entropy for n items (uniform distribution)."""
    return np.log(n)


def normalized_entropy(scores: np.ndarray, temperature: float = 1.0) -> float:
    """
    Entropy normalized to [0, 1].
    0 = all attention on one node (fully locked).
    1 = uniform attention (fully diverse).
    """
    n = len(scores)
    if n <= 1:
        return 0.0
    h = score_entropy(scores, temperature)
    h_max = max_entropy(n)
    return h / h_max


def top_k_concentration(scores: np.ndarray, k: int = 5) -> float:
    """
    Fraction of total probability mass in top-k.
    High concentration = locked on few nodes.
    
    Returns value in [k/n, 1.0].
    """
    probs = softmax(scores)
    top_k_probs = np.sort(probs)[::-1][:k]
    return np.sum(top_k_probs)


def diversity_at_k(scores: np.ndarray, embeddings: np.ndarray, 
                   k: int = 5, cluster_threshold: float = 0.85) -> int:
    """
    Count unique content clusters in top-k retrieved nodes.
    
    If top-k nodes are all near-duplicates (high pairwise similarity),
    diversity is low → sign of lock.
    
    Args:
        scores: Retrieval scores.
        embeddings: Node embeddings (n, d).
        k: Number of top nodes to consider.
        cluster_threshold: Similarity above which two nodes are "same cluster".
    
    Returns:
        Number of unique clusters in top-k.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    top_idx = np.argsort(scores)[::-1][:k]
    top_embs = embeddings[top_idx]
    
    # Simple greedy clustering
    clusters = []
    for i in range(len(top_idx)):
        assigned = False
        for cluster in clusters:
            # Check similarity to cluster representative
            rep_emb = top_embs[cluster[0]].reshape(1, -1)
            sim = cosine_similarity(top_embs[i].reshape(1, -1), rep_emb)[0, 0]
            if sim > cluster_threshold:
                cluster.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])
    
    return len(clusters)


def detect_lock(entropies: List[float], threshold: float = 0.3) -> Dict:
    """
    Detect attention lock from a sequence of entropy values across queries.
    
    A retriever is "locked" if:
    1. Mean entropy is low (attending to narrow set)
    2. Entropy variance is low (same narrow set regardless of query)
    
    Args:
        entropies: List of normalized entropy values, one per query.
        threshold: Normalized entropy below which a single query is "locked".
    
    Returns:
        Dict with lock detection results.
    """
    entropies = np.array(entropies)
    
    mean_h = np.mean(entropies)
    std_h = np.std(entropies)
    lock_ratio = np.mean(entropies < threshold)
    
    # Lock classification
    if lock_ratio > 0.7 and std_h < 0.1:
        status = "LOCKED"
        description = "Retriever consistently attends to narrow node set regardless of query."
    elif lock_ratio > 0.4:
        status = "PARTIAL_LOCK"
        description = "Retriever frequently concentrates attention. May be over-relying on one signal."
    elif mean_h > 0.7 and std_h > 0.15:
        status = "HEALTHY"
        description = "Attention adapts to queries with good diversity."
    else:
        status = "NORMAL"
        description = "No strong lock detected. Attention is reasonably distributed."
    
    return {
        "status": status,
        "description": description,
        "mean_entropy": float(mean_h),
        "std_entropy": float(std_h),
        "lock_ratio": float(lock_ratio),
        "n_queries": len(entropies),
        "threshold": threshold,
    }


def run_entropy_diagnostic(
    retriever_fn,
    queries: List[Dict],
    nodes: List[Dict],
    embeddings: np.ndarray,
    feature_builder,
    k: int = 5,
    temperature: float = 1.0,
    lock_threshold: float = 0.3,
) -> Dict:
    """
    Run full entropy diagnostic on a retriever across multiple queries.
    
    Args:
        retriever_fn: Function(features) -> scores array.
        queries: List of query dicts with 'text' field.
        nodes: List of node dicts.
        embeddings: Node embeddings (n, d).
        feature_builder: Function(query_text) -> feature_matrix.
        k: Top-k for diversity calculation.
        temperature: Softmax temperature for entropy.
        lock_threshold: Threshold for lock detection.
    
    Returns:
        Full diagnostic report.
    """
    entropies = []
    concentrations = []
    diversities = []
    
    for query in queries:
        features = feature_builder(query["text"])
        scores = retriever_fn(features)
        
        h_norm = normalized_entropy(scores, temperature)
        conc = top_k_concentration(scores, k)
        div = diversity_at_k(scores, embeddings, k)
        
        entropies.append(h_norm)
        concentrations.append(conc)
        diversities.append(div)
    
    lock_result = detect_lock(entropies, lock_threshold)
    
    return {
        **lock_result,
        "mean_concentration": float(np.mean(concentrations)),
        "mean_diversity_at_k": float(np.mean(diversities)),
        "per_query_entropy": entropies,
        "per_query_concentration": concentrations,
        "per_query_diversity": diversities,
    }


def compare_retrievers_entropy(
    retriever_scores: Dict[str, List[np.ndarray]],
    n_nodes: int,
    temperature: float = 1.0,
) -> Dict[str, Dict]:
    """
    Compare entropy profiles across multiple retrievers.
    
    Args:
        retriever_scores: {retriever_name: [scores_per_query, ...]}.
        n_nodes: Number of nodes in the memory store.
        temperature: Softmax temperature.
    
    Returns:
        Comparative diagnostic per retriever.
    """
    results = {}
    
    for name, scores_list in retriever_scores.items():
        entropies = [normalized_entropy(s, temperature) for s in scores_list]
        lock = detect_lock(entropies)
        results[name] = {
            **lock,
            "raw_entropies": entropies,
        }
    
    # Cross-retriever comparison
    healthy_retrievers = [n for n, r in results.items() if r["status"] in ("HEALTHY", "NORMAL")]
    locked_retrievers = [n for n, r in results.items() if r["status"] in ("LOCKED", "PARTIAL_LOCK")]
    
    return {
        "per_retriever": results,
        "summary": {
            "healthy": healthy_retrievers,
            "locked": locked_retrievers,
            "recommendation": _generate_recommendation(results),
        }
    }


def _generate_recommendation(results: Dict[str, Dict]) -> str:
    """Generate actionable recommendation from entropy diagnostic."""
    locked = [n for n, r in results.items() if r["status"] == "LOCKED"]
    partial = [n for n, r in results.items() if r["status"] == "PARTIAL_LOCK"]
    
    if locked:
        return (f"Retrievers {locked} show attention lock. Consider: "
                f"(1) adding diversity penalty, (2) increasing temperature, "
                f"(3) checking if one feature dominates theta.")
    elif partial:
        return (f"Retrievers {partial} show partial lock on some queries. "
                f"Monitor per-family entropy to identify which query types trigger lock.")
    else:
        return "All retrievers show healthy attention diversity. No intervention needed."
