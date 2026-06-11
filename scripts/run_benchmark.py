"""
Main benchmark script: run all retrievers, compute metrics, output results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import pandas as pd
from generator import generate_world, save_world
from similarity import ToySimilarity
from estimators import build_feature_matrix
from retrievers import RETRIEVERS, retrieve_mga_gate, retrieve_mga_linear, top_k
from metrics import evaluate_retrieval
from learn import learn_theta_gate, learn_theta_linear


def run_single_world(seed: int, k: int = 5, verbose: bool = False):
    """Run full benchmark on one synthetic world."""
    
    # Generate world
    world = generate_world(seed=seed)
    nodes = world["nodes"]
    queries = world["queries"]
    oracle = world["oracle"]
    
    if verbose:
        print(f"  World {seed}: {len(nodes)} nodes, {len(queries)} queries")
    
    # Build similarity
    sim_module = ToySimilarity()
    texts = [n["content"] for n in nodes]
    node_embeddings = sim_module.fit(texts)
    sim_matrix = sim_module.pairwise()
    
    # Use last timestamp as query time
    query_time = nodes[-1]["timestamp"]
    
    # Goal text (simple: concatenate all non-noise, non-stale node texts as "project context")
    goal_text = " ".join(n["content"] for n in nodes 
                        if n["id"] not in oracle["noise_nodes"]
                        and n["event_type"] in ("state_fact", "revise_fact"))[:500]
    
    # Split queries: first 40% train, next 20% dev, last 40% test
    n_q = len(queries)
    train_end = int(n_q * 0.4)
    dev_end = int(n_q * 0.6)
    train_queries = queries[:train_end]
    test_queries = queries[dev_end:]
    
    # Build features for all nodes
    node_ids = [n["id"] for n in nodes]
    stale_set = set(oracle["stale_nodes"])
    noise_set = set(oracle["noise_nodes"])
    
    # Learn theta on train queries
    train_features_list = []
    train_labels_list = []
    
    for q in train_queries:
        gold_set = set(q.get("gold_nodes", []))
        sim_scores = sim_module.query_similarity(q["text"])
        features = build_feature_matrix(
            nodes, query_time, sim_scores, sim_matrix,
            node_embeddings, goal_text, sim_module
        )
        labels = np.array([1.0 if nid in gold_set else 0.0 for nid in node_ids])
        train_features_list.append(features)
        train_labels_list.append(labels)
    
    # Learn
    theta_gate = learn_theta_gate(train_features_list, train_labels_list)
    theta_linear = learn_theta_linear(train_features_list, train_labels_list)
    
    # Evaluate on test queries
    results = []
    
    for q in test_queries:
        gold_set = set(q.get("gold_nodes", []))
        if not gold_set:
            continue
        
        sim_scores = sim_module.query_similarity(q["text"])
        features = build_feature_matrix(
            nodes, query_time, sim_scores, sim_matrix,
            node_embeddings, goal_text, sim_module
        )
        
        for retriever_name, retriever_fn in RETRIEVERS.items():
            if retriever_name == "mga_gate":
                scores = retriever_fn(features, theta_gate)
            elif retriever_name == "mga_linear":
                scores = retriever_fn(features, theta_linear)
            else:
                scores = retriever_fn(features)
            
            top_indices = top_k(scores, k)
            retrieved_ids = [node_ids[i] for i in top_indices]
            
            metrics = evaluate_retrieval(retrieved_ids, gold_set, stale_set, noise_set, k)
            metrics["retriever"] = retriever_name
            metrics["query_id"] = q["id"]
            metrics["query_family"] = q.get("family", "unknown")
            metrics["seed"] = seed
            results.append(metrics)
    
    return results, theta_gate, theta_linear


def run_full_benchmark(n_seeds: int = 5, k: int = 5):
    """Run benchmark across multiple seeds."""
    
    print("=" * 70)
    print("  MGA BENCHMARK — Memory-Gravity Attention vs Baselines")
    print("  (Existential Attention applied to memory retrieval)")
    print("=" * 70)
    
    all_results = []
    all_theta_gate = []
    all_theta_linear = []
    
    for seed in range(n_seeds):
        results, theta_g, theta_l = run_single_world(seed, k, verbose=True)
        all_results.extend(results)
        all_theta_gate.append(theta_g)
        all_theta_linear.append(theta_l)
    
    # Aggregate
    df = pd.DataFrame(all_results)
    
    print(f"\n  Total queries evaluated: {len(df)}")
    print(f"  Seeds: {n_seeds}, k={k}")
    
    # Summary table
    print(f"\n{'='*70}")
    print(f"  RESULTS: Retriever Comparison")
    print(f"{'='*70}")
    
    summary = df.groupby("retriever").agg({
        "recall@k": ["mean", "std"],
        "precision@k": ["mean", "std"],
        "ndcg@k": ["mean", "std"],
        "stale@k": ["mean", "std"],
        "noise@k": ["mean", "std"],
    }).round(4)
    
    print(f"\n  {'Retriever':<12} {'Recall@k':>10} {'nDCG@k':>10} {'Stale@k':>10} {'Noise@k':>10}")
    print(f"  {'-'*52}")
    
    for retriever in ["recency", "sim", "ga", "mga_gate", "mga_linear"]:
        r_df = df[df["retriever"] == retriever]
        if len(r_df) == 0:
            continue
        recall = r_df["recall@k"].mean()
        ndcg = r_df["ndcg@k"].mean()
        stale = r_df["stale@k"].mean()
        noise = r_df["noise@k"].mean()
        print(f"  {retriever:<12} {recall:>10.4f} {ndcg:>10.4f} {stale:>10.4f} {noise:>10.4f}")
    
    # Per-family breakdown
    print(f"\n{'='*70}")
    print(f"  PER-FAMILY BREAKDOWN (Recall@k)")
    print(f"{'='*70}")
    
    families = df["query_family"].unique()
    print(f"\n  {'Family':<15}", end="")
    for r in ["recency", "sim", "ga", "mga_gate", "mga_linear"]:
        print(f" {r:>10}", end="")
    print()
    print(f"  {'-'*67}")
    
    for fam in sorted(families):
        fam_df = df[df["query_family"] == fam]
        print(f"  {fam:<15}", end="")
        for r in ["recency", "sim", "ga", "mga_gate", "mga_linear"]:
            r_fam = fam_df[fam_df["retriever"] == r]
            if len(r_fam) > 0:
                print(f" {r_fam['recall@k'].mean():>10.4f}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        print()
    
    # Learned theta
    print(f"\n{'='*70}")
    print(f"  LEARNED THETA (MGA Gate)")
    print(f"{'='*70}")
    feature_names = ["recency", "frequency", "unresolved", "utility", "weight", "goal_rel"]
    mean_theta = np.mean(all_theta_gate, axis=0)
    std_theta = np.std(all_theta_gate, axis=0)
    print(f"\n  {'Feature':<15} {'Mean coef':>12} {'Std':>10}")
    print(f"  {'-'*37}")
    for name, mean, std in zip(feature_names, mean_theta, std_theta):
        print(f"  {name:<15} {mean:>12.4f} {std:>10.4f}")
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    print(f"\n  Results saved to outputs/results.csv")
    
    return df


if __name__ == "__main__":
    run_full_benchmark(n_seeds=5, k=5)
