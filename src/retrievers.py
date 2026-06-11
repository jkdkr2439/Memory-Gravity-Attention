"""
Retrievers (§5 of spec)

5 retrieval methods:
1. recency: rank by recency only
2. sim: rank by similarity only
3. ga: Generative-Agents style (recency + importance + similarity)
4. mga_gate: similarity × (1 + sigmoid(theta^T features)) — THE ONE TO TEST
5. mga_linear: theta^T features (ablation)
"""

import numpy as np
from scipy.special import expit  # sigmoid


def retrieve_recency(features: np.ndarray, k: int = 5) -> np.ndarray:
    """Rank by recency only (column 1)."""
    scores = features[:, 1]  # recency
    return scores


def retrieve_similarity(features: np.ndarray, k: int = 5) -> np.ndarray:
    """Rank by similarity only (column 0)."""
    scores = features[:, 0]  # similarity
    return scores


def retrieve_ga(features: np.ndarray, k: int = 5, alpha=1.0, beta=1.0, gamma=1.0) -> np.ndarray:
    """
    Generative-Agents baseline (Park et al. 2023):
    score = alpha * recency + beta * importance + gamma * similarity
    importance proxied by weight feature.
    """
    sim = features[:, 0]
    recency = features[:, 1]
    importance = features[:, 5]  # weight as proxy
    scores = alpha * recency + beta * importance + gamma * sim
    return scores


def retrieve_mga_gate(features: np.ndarray, theta: np.ndarray, k: int = 5) -> np.ndarray:
    """
    MGA Gate (default):
    score = sim × (1 + sigmoid(theta^T f_persistent))
    
    f_persistent = all features EXCEPT similarity (cols 1-6)
    This is the key: similarity gates whether a node can be retrieved,
    but persistent importance AMPLIFIES relevant nodes.
    
    A node with low similarity cannot be dragged in by importance alone.
    A node with high similarity AND high importance gets boosted.
    """
    sim = features[:, 0]
    f_persistent = features[:, 1:]  # everything except sim
    gate = expit(f_persistent @ theta)  # sigmoid(theta^T f)
    scores = sim * (1.0 + gate)
    return scores


def retrieve_mga_linear(features: np.ndarray, theta: np.ndarray, k: int = 5) -> np.ndarray:
    """
    MGA Linear (ablation):
    score = theta^T features
    No gate structure. Tests whether the gate matters.
    """
    scores = features @ theta
    return scores


def top_k(scores: np.ndarray, k: int = 5) -> np.ndarray:
    """Return indices of top-k scores."""
    return np.argsort(scores)[::-1][:k]


RETRIEVERS = {
    "recency": retrieve_recency,
    "sim": retrieve_similarity,
    "ga": retrieve_ga,
    "mga_gate": retrieve_mga_gate,
    "mga_linear": retrieve_mga_linear,
}
