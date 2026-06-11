"""
Signal Estimators (§4 of spec)

Each estimator: (log, node, t_query) -> value in [0,1]
All are heuristics. They never see oracle/gold labels.
"""

import numpy as np
from datetime import datetime
from typing import List, Dict


def estimate_recency(nodes: List[Dict], query_time: str, tau_r: float = 5.0) -> np.ndarray:
    """
    Recency: exp(-(t_query - t_node) / tau_r)
    tau_r in days.
    """
    t_q = datetime.fromisoformat(query_time)
    recency = np.zeros(len(nodes))
    for i, node in enumerate(nodes):
        t_n = datetime.fromisoformat(node["timestamp"])
        delta_days = (t_q - t_n).total_seconds() / 86400.0
        recency[i] = np.exp(-delta_days / tau_r)
    return recency


def estimate_frequency(nodes: List[Dict], similarity_matrix: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """
    Frequency: how often this node's content is re-mentioned.
    Uses similarity matrix to find near-duplicates.
    """
    n = len(nodes)
    mentions = np.zeros(n)
    for i in range(n):
        # Count how many OTHER nodes are highly similar (re-mentions)
        mentions[i] = np.sum(similarity_matrix[i] > threshold) - 1  # exclude self
    
    if mentions.max() > 0:
        freq = np.log1p(mentions) / np.log1p(mentions.max())
    else:
        freq = np.zeros(n)
    return freq


def estimate_unresolved(nodes: List[Dict], similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Unresolvedness: is this node a task that hasn't been closed?
    Heuristic: task cues + no closure match found.
    """
    task_cues = ["need to", "still", "todo", "pending", "waiting", "not done", "remains", "fix", "resolve"]
    closure_cues = ["done", "fixed", "completed", "finished", "resolved", "cancelled", "closed"]
    
    n = len(nodes)
    unresolved = np.zeros(n)
    
    for i, node in enumerate(nodes):
        content_lower = node["content"].lower()
        
        # Check if task-like
        is_task = any(cue in content_lower for cue in task_cues)
        if not is_task:
            continue
        
        # Check if any later node closes it
        has_closure = False
        for j in range(i + 1, n):
            other_lower = nodes[j]["content"].lower()
            has_closure_cue = any(cue in other_lower for cue in closure_cues)
            is_similar = similarity_matrix[i, j] > 0.5
            if has_closure_cue and is_similar:
                has_closure = True
                break
        
        unresolved[i] = 0.0 if has_closure else 1.0
    
    return unresolved


def estimate_goal_relevance(nodes: List[Dict], node_embeddings: np.ndarray, 
                            goal_text: str, sim_module) -> np.ndarray:
    """
    Goal relevance: cosine similarity between node and active project description.
    """
    goal_emb = sim_module.encode(goal_text).reshape(1, -1)
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(goal_emb, node_embeddings)[0]
    # Min-max normalize
    if sims.max() > sims.min():
        sims = (sims - sims.min()) / (sims.max() - sims.min())
    return sims


def build_feature_matrix(nodes, query_time, sim_scores, similarity_matrix, 
                         node_embeddings, goal_text, sim_module,
                         utility=None, weight=None) -> np.ndarray:
    """
    Build the full feature matrix: [sim, recency, freq, unresolved, utility, weight, goal_rel]
    All in [0,1].
    """
    n = len(nodes)
    
    recency = estimate_recency(nodes, query_time)
    frequency = estimate_frequency(nodes, similarity_matrix)
    unresolved = estimate_unresolved(nodes, similarity_matrix)
    goal_rel = estimate_goal_relevance(nodes, node_embeddings, goal_text, sim_module)
    
    if utility is None:
        utility = np.full(n, 0.5)  # uninformative prior
    if weight is None:
        weight = np.full(n, 0.5)
    
    # Normalize sim_scores to [0,1]
    sim_norm = sim_scores.copy()
    if sim_norm.max() > sim_norm.min():
        sim_norm = (sim_norm - sim_norm.min()) / (sim_norm.max() - sim_norm.min())
    
    features = np.column_stack([
        sim_norm,       # 0: similarity
        recency,        # 1: recency
        frequency,      # 2: frequency
        unresolved,     # 3: unresolvedness
        utility,        # 4: utility
        weight,         # 5: persistent weight
        goal_rel,       # 6: goal relevance
    ])
    
    return features
