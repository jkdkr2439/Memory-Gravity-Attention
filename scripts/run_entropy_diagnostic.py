"""
Run entropy diagnostic on all retrievers.
Shows which retrievers are "locked" (attending to narrow node sets)
vs which have healthy, query-adaptive attention.

Usage: python scripts/run_entropy_diagnostic.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from generator import generate_world
from similarity import ToySimilarity
from estimators import build_feature_matrix
from retrievers import RETRIEVERS, top_k
from learn import learn_theta_gate, learn_theta_linear
from entropy_diagnostic import (
    normalized_entropy, top_k_concentration,
    detect_lock, compare_retrievers_entropy,
)


def run_diagnostic(n_seeds=5, k=5):
    print("=" * 70)
    print("  ATTENTION-ENTROPY DIAGNOSTIC")
    print("  Detecting retriever lock across synthetic worlds")
    print("=" * 70)

    all_scores = {name: [] for name in RETRIEVERS}

    for seed in range(n_seeds):
        world = generate_world(seed=seed)
        nodes = world["nodes"]
        queries = world["queries"]
        oracle = world["oracle"]

        # Build similarity
        sim_module = ToySimilarity()
        texts = [n["content"] for n in nodes]
        node_embeddings = sim_module.fit(texts)
        sim_matrix = sim_module.pairwise()

        query_time = nodes[-1]["timestamp"]
        stale_set = set(oracle["stale_nodes"])
        noise_set = set(oracle["noise_nodes"])
        goal_text = " ".join(
            n["content"] for n in nodes
            if n["id"] not in noise_set and n["event_type"] in ("state_fact", "revise_fact")
        )[:500]
        node_ids = [n["id"] for n in nodes]

        # Train theta
        n_q = len(queries)
        train_end = max(int(n_q * 0.4), 2)
        train_qs = queries[:train_end]
        test_qs = queries[train_end:]

        train_features_list = []
        train_labels_list = []
        for q in train_qs:
            gold_set = set(q.get("gold_nodes", []))
            sim_scores = sim_module.query_similarity(q["text"])
            features = build_feature_matrix(
                nodes, query_time, sim_scores, sim_matrix,
                node_embeddings, goal_text, sim_module
            )
            labels = np.array([1.0 if nid in gold_set else 0.0 for nid in node_ids])
            train_features_list.append(features)
            train_labels_list.append(labels)

        theta_gate = learn_theta_gate(train_features_list, train_labels_list)
        theta_linear = learn_theta_linear(train_features_list, train_labels_list)

        # Collect scores for each retriever on test queries
        for q in test_qs:
            sim_scores = sim_module.query_similarity(q["text"])
            features = build_feature_matrix(
                nodes, query_time, sim_scores, sim_matrix,
                node_embeddings, goal_text, sim_module
            )

            for name, retriever_fn in RETRIEVERS.items():
                if name == "mga_gate":
                    scores = retriever_fn(features, theta_gate)
                elif name == "mga_linear":
                    scores = retriever_fn(features, theta_linear)
                else:
                    scores = retriever_fn(features)
                all_scores[name].append(scores)

    # Run comparative diagnostic
    print(f"\n  Analyzed {n_seeds} worlds, {sum(len(v) for v in all_scores.values()) // len(RETRIEVERS)} queries per retriever")

    result = compare_retrievers_entropy(all_scores, n_nodes=len(nodes))

    print(f"\n  {'Retriever':<12} {'Status':<15} {'Mean H':>8} {'Std H':>8} {'Lock%':>8} {'Concentration':>14}")
    print(f"  {'-' * 67}")

    for name in ["recency", "sim", "ga", "mga_gate", "mga_linear"]:
        r = result["per_retriever"][name]
        # Compute mean concentration
        concentrations = [top_k_concentration(s, k) for s in all_scores[name]]
        print(f"  {name:<12} {r['status']:<15} {r['mean_entropy']:>8.4f} {r['std_entropy']:>8.4f} "
              f"{r['lock_ratio']*100:>7.1f}% {np.mean(concentrations):>13.4f}")

    print(f"\n  SUMMARY:")
    print(f"    Healthy: {result['summary']['healthy']}")
    print(f"    Locked:  {result['summary']['locked']}")
    print(f"\n  RECOMMENDATION:")
    print(f"    {result['summary']['recommendation']}")

    # Per-retriever entropy histogram
    print(f"\n  ENTROPY DISTRIBUTION (quartiles):")
    print(f"  {'Retriever':<12} {'Q1':>8} {'Median':>8} {'Q3':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-' * 52}")
    for name in ["recency", "sim", "ga", "mga_gate", "mga_linear"]:
        ents = result["per_retriever"][name]["raw_entropies"]
        q1, med, q3 = np.percentile(ents, [25, 50, 75])
        print(f"  {name:<12} {q1:>8.4f} {med:>8.4f} {q3:>8.4f} {min(ents):>8.4f} {max(ents):>8.4f}")

    print(f"\n{'=' * 70}")
    print("  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_diagnostic(n_seeds=5, k=5)
