"""
Metrics (§6 of spec)
Recall@k, Precision@k, nDCG@k, Stale@k, Noise@k
"""

import numpy as np
from typing import List, Set


def recall_at_k(retrieved: List[str], gold: Set[str], k: int = 5) -> float:
    """Fraction of gold nodes that appear in top-k retrieved."""
    if not gold:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & gold) / len(gold)


def precision_at_k(retrieved: List[str], gold: Set[str], k: int = 5) -> float:
    """Fraction of top-k that are gold."""
    retrieved_k = set(retrieved[:k])
    if k == 0:
        return 0.0
    return len(retrieved_k & gold) / k


def ndcg_at_k(retrieved: List[str], gold: Set[str], k: int = 5) -> float:
    """Normalized DCG@k."""
    dcg = 0.0
    for i, node_id in enumerate(retrieved[:k]):
        if node_id in gold:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because rank starts at 1
    
    # Ideal DCG
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
    
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def stale_at_k(retrieved: List[str], stale_nodes: Set[str], k: int = 5) -> float:
    """Fraction of top-k that are stale (superseded)."""
    retrieved_k = set(retrieved[:k])
    if k == 0:
        return 0.0
    return len(retrieved_k & stale_nodes) / k


def noise_at_k(retrieved: List[str], noise_nodes: Set[str], k: int = 5) -> float:
    """Fraction of top-k that are noise (chatter)."""
    retrieved_k = set(retrieved[:k])
    if k == 0:
        return 0.0
    return len(retrieved_k & noise_nodes) / k


def evaluate_retrieval(retrieved_ids: List[str], gold_ids: Set[str], 
                       stale_ids: Set[str], noise_ids: Set[str], k: int = 5) -> dict:
    """Full evaluation metrics for one query."""
    return {
        "recall@k": recall_at_k(retrieved_ids, gold_ids, k),
        "precision@k": precision_at_k(retrieved_ids, gold_ids, k),
        "ndcg@k": ndcg_at_k(retrieved_ids, gold_ids, k),
        "stale@k": stale_at_k(retrieved_ids, stale_ids, k),
        "noise@k": noise_at_k(retrieved_ids, noise_ids, k),
    }
