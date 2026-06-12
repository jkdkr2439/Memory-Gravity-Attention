"""
MGA on LoCoMo-MC10 Benchmark (derived from Snap Research LoCoMo, 2024)
"Evaluating Very Long-Term Conversational Memory of LLM Agents"

LoCoMo-MC10: 1,986 items, multi-session conversations, 5 question types.
Fields: haystack_sessions (full conv), question, answer, question_type, 
        haystack_session_summaries, haystack_session_datetimes.

We convert each conversation into MGA memory nodes, then test retrieval:
- Can MGA retrieve the correct session/turns that answer the question?

Run: modal run scripts/modal_locomo_benchmark.py
"""

import modal

app = modal.App("mga-locomo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "scikit-learn", "sentence-transformers", 
                 "torch", "scipy", "datasets")
)


@app.function(image=image, gpu="T4", timeout=1200)
def run_locomo_benchmark():
    import numpy as np
    import pandas as pd
    import random
    from datetime import datetime, timedelta
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from scipy.special import expit
    from scipy import stats
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    print("=" * 70)
    print("  MGA on LoCoMo-MC10 BENCHMARK")
    print("=" * 70)

    print("\nLoading sentence-transformers...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading LoCoMo-MC10 dataset from HuggingFace...")
    ds = load_dataset("Percena/locomo-mc10", split="train")
    print(f"Loaded: {len(ds)} items")
    print(f"Columns: {ds.column_names}")
    print(f"Question types: {set(ds['question_type'])}")

    # ====================================================================
    # PROCESS LoCoMo-MC10 INTO MGA FORMAT
    # ====================================================================
    print("\nProcessing LoCoMo conversations into MGA worlds...")

    def process_locomo_item(item, idx):
        """
        Convert LoCoMo-MC10 item to MGA world format.
        
        Key fields:
        - haystack_sessions: list[list[dict]] — full conversation data per session
        - haystack_session_summaries: list[str] — session summaries
        - haystack_session_datetimes: list — timestamps per session
        - question: str — the memory query
        - answer: str — correct answer text
        - question_type: str — single_hop, multi_hop, temporal_reasoning, etc.
        """
        sessions = item.get("haystack_sessions", [])
        question = item.get("question", "")
        answer = str(item.get("answer", ""))
        q_type = item.get("question_type", "unknown")
        summaries = item.get("haystack_session_summaries", [])
        datetimes = item.get("haystack_session_datetimes", [])

        if not sessions or not question or not answer:
            return None

        # Build nodes from conversation turns
        nodes = []
        base_time = datetime(2024, 1, 1, 9, 0, 0)

        for sess_idx, session in enumerate(sessions):
            if not isinstance(session, list):
                continue

            # Session time from datetimes if available
            if datetimes and sess_idx < len(datetimes):
                try:
                    sess_time = datetime.fromisoformat(str(datetimes[sess_idx]).replace("Z", "+00:00").split("+")[0])
                except (ValueError, TypeError):
                    sess_time = base_time + timedelta(days=sess_idx * 3)
            else:
                sess_time = base_time + timedelta(days=sess_idx * 3)

            for turn_idx, turn in enumerate(session):
                # Extract content from turn
                if isinstance(turn, dict):
                    content = turn.get("text", turn.get("content", turn.get("utterance", "")))
                    speaker = turn.get("speaker", turn.get("from", turn.get("role", f"speaker_{turn_idx % 2}")))
                elif isinstance(turn, str):
                    content = turn
                    speaker = f"speaker_{turn_idx % 2}"
                else:
                    continue

                if not content or len(str(content)) < 5:
                    continue

                content = str(content)[:600]
                ts = sess_time + timedelta(minutes=turn_idx * 2)

                # Classify node type heuristically
                content_lower = content.lower()
                if any(w in content_lower for w in ["remember", "always", "rule", "must", "important", "note"]):
                    ntype = "state_fact"
                elif any(w in content_lower for w in ["todo", "need to", "still", "pending", "haven't", "should"]):
                    ntype = "open_task"
                elif any(w in content_lower for w in ["actually", "change", "instead", "correction", "update"]):
                    ntype = "revision"
                elif len(content) < 25 or any(w in content_lower for w in ["haha", "lol", "ok", "sure", "yeah", "thanks", "bye", "hey"]):
                    ntype = "chatter"
                else:
                    ntype = "content"

                nodes.append({
                    "id": f"n{len(nodes):04d}",
                    "content": content,
                    "ts": ts.isoformat(),
                    "sess": sess_idx,
                    "type": ntype,
                    "speaker": str(speaker),
                })

        if len(nodes) < 10:
            return None

        # Identify gold nodes: which nodes contain information needed to answer?
        gold_nodes = []
        answer_lower = answer.lower()
        answer_words = [w for w in answer_lower.split() if len(w) > 3][:12]

        if answer_lower in ["not answerable", "unanswerable", "n/a"]:
            # Adversarial — no gold nodes (any retrieval is wrong)
            # Skip these for retrieval evaluation
            return None

        for n in nodes:
            n_lower = n["content"].lower()
            # Check word overlap
            matches = sum(1 for w in answer_words if w in n_lower)
            if matches >= max(2, len(answer_words) // 4):
                gold_nodes.append(n["id"])

        # If no gold nodes found by word overlap, try sentence similarity later
        # For now, require at least 1 gold node
        if not gold_nodes:
            return None

        # Limit gold to top 5 most relevant
        gold_nodes = gold_nodes[:5]

        # Oracle labels
        noise = set(n["id"] for n in nodes if n["type"] == "chatter")
        stale = set()  # No explicit stale in LoCoMo

        # Map question type to MGA family
        family_map = {
            "single_hop": "needle",
            "multi_hop": "multi_hop",
            "temporal_reasoning": "temporal",
            "open_domain": "constraint",
            "adversarial": "adversarial",
        }
        family = family_map.get(q_type, "general")

        query = {
            "id": f"q_{idx}",
            "fam": family,
            "text": question,
            "gold": gold_nodes,
        }

        n_sessions = max(n["sess"] for n in nodes) + 1
        return {
            "nodes": nodes,
            "queries": [query],
            "stale": stale,
            "noise": noise,
            "n_sessions": n_sessions,
        }

    # Process all items
    worlds = []
    skipped = {"no_sessions": 0, "too_short": 0, "no_gold": 0, "adversarial": 0}

    for i in range(len(ds)):
        if len(worlds) >= 200:  # Cap at 200 conversations
            break
        try:
            world = process_locomo_item(ds[i], i)
            if world:
                worlds.append(world)
                if len(worlds) <= 3:
                    print(f"  Sample {len(worlds)}: {len(world['nodes'])} nodes, "
                          f"{world['n_sessions']} sessions, "
                          f"family={world['queries'][0]['fam']}, "
                          f"gold={len(world['queries'][0]['gold'])}")
            else:
                # Count skip reasons
                item = ds[i]
                if not item.get("haystack_sessions"):
                    skipped["no_sessions"] += 1
                elif str(item.get("answer", "")).lower() in ["not answerable", "unanswerable"]:
                    skipped["adversarial"] += 1
                else:
                    skipped["no_gold"] += 1
        except Exception as e:
            if len(worlds) < 5:
                print(f"  Error on item {i}: {e}")
            continue

    print(f"\nProcessed: {len(worlds)} valid worlds")
    print(f"Skipped: {skipped}")

    if len(worlds) < 10:
        print("ERROR: Not enough valid items. Inspect data format.")
        # Debug: show raw sample
        sample = ds[0]
        print(f"\nSample item keys: {list(sample.keys())}")
        for k in sample.keys():
            v = sample[k]
            if isinstance(v, str):
                print(f"  {k}: {v[:100]}")
            elif isinstance(v, list):
                print(f"  {k}: list[{len(v)}] first={str(v[0])[:100] if v else 'empty'}")
            else:
                print(f"  {k}: {type(v)} = {str(v)[:100]}")
        return {"error": "insufficient valid data", "n_worlds": len(worlds), "skipped": skipped}

    # ====================================================================
    # EMBED
    # ====================================================================
    print("\nEmbedding all nodes...")
    for wi, w in enumerate(worlds):
        texts = [n["content"] for n in w["nodes"]]
        w["embs"] = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        w["sim_mat"] = cosine_similarity(w["embs"])
        # Goal text: concatenate substantive content
        w["goal"] = " ".join(
            n["content"] for n in w["nodes"]
            if n["type"] in ("state_fact", "content") and n["id"] not in w["noise"]
        )[:800]
        if (wi + 1) % 50 == 0:
            print(f"  Embedded {wi + 1}/{len(worlds)} worlds")
    print(f"  Done embedding {len(worlds)} worlds")

    # ====================================================================
    # FEATURES & RETRIEVERS
    # ====================================================================
    def compute_features(nodes, query_text, embeddings, sim_matrix, embed_model, goal_text):
        n = len(nodes)
        q_emb = embed_model.encode([query_text], normalize_embeddings=True)
        sim_scores = cosine_similarity(q_emb, embeddings)[0]

        t_q = datetime.fromisoformat(nodes[-1]["ts"])
        recency = np.array([
            np.exp(-(t_q - datetime.fromisoformat(nd["ts"])).total_seconds() / (5 * 86400))
            for nd in nodes
        ])

        freq_counts = np.array([(sim_matrix[i] > 0.75).sum() - 1 for i in range(n)], dtype=float)
        freq = np.log1p(freq_counts) / max(np.log1p(freq_counts.max()), 1e-8)

        task_cues = ["need to", "still", "todo", "pending", "waiting", "should", "haven't"]
        closure_cues = ["done", "fixed", "completed", "finished", "resolved"]
        unresolved = np.zeros(n)
        for i, nd in enumerate(nodes):
            cl = nd["content"].lower()
            if any(c in cl for c in task_cues):
                has_close = any(
                    any(c in nodes[j]["content"].lower() for c in closure_cues) and sim_matrix[i, j] > 0.4
                    for j in range(i + 1, min(i + 50, n))  # limit search window
                )
                unresolved[i] = 0.0 if has_close else 1.0

        g_emb = embed_model.encode([goal_text], normalize_embeddings=True)
        goal_rel = cosine_similarity(g_emb, embeddings)[0]
        gr_min, gr_max = goal_rel.min(), goal_rel.max()
        goal_rel = (goal_rel - gr_min) / max(gr_max - gr_min, 1e-8)

        s_min, s_max = sim_scores.min(), sim_scores.max()
        sim_norm = (sim_scores - s_min) / max(s_max - s_min, 1e-8)

        return np.column_stack([sim_norm, recency, freq, unresolved, np.full(n, .5), np.full(n, .5), goal_rel])

    def retrieve(features, method, theta_g=None, theta_l=None):
        sim = features[:, 0]
        if method == "recency":
            return features[:, 1]
        elif method == "sim":
            return sim
        elif method == "ga":
            return sim + features[:, 1] + features[:, 5]
        elif method == "mga_gate":
            return sim * (1.0 + expit(features[:, 1:] @ theta_g))
        elif method == "mga_linear":
            return features @ theta_l
        return sim

    # ====================================================================
    # K-FOLD CV
    # ====================================================================
    METHODS = ["recency", "sim", "ga", "mga_gate", "mga_linear"]
    K = 5

    all_queries = []
    for wi, w in enumerate(worlds):
        for q in w["queries"]:
            all_queries.append({"world_idx": wi, **q})

    print(f"\nTotal queries: {len(all_queries)}")
    print(f"Running 5-fold CV...")

    n_folds = min(5, len(all_queries))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []
    thetas = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(all_queries)))):
        # Learn theta on train
        X_tr, y_tr = [], []
        for qi in train_idx:
            q = all_queries[qi]
            w = worlds[q["world_idx"]]
            gold = set(q["gold"])
            feats = compute_features(w["nodes"], q["text"], w["embs"], w["sim_mat"], embed_model, w["goal"])
            labels = np.array([1.0 if n["id"] in gold else 0.0 for n in w["nodes"]])
            X_tr.append(feats[:, 1:])
            y_tr.append(labels)

        X_tr = np.vstack(X_tr)
        y_tr = np.concatenate(y_tr)

        if len(np.unique(y_tr)) >= 2:
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X_tr, y_tr)
            theta_g = clf.coef_[0]
        else:
            theta_g = np.ones(6) * 0.5
        thetas.append(theta_g)

        # Linear theta (all 7 features)
        X_lin_parts, y_lin_parts = [], []
        for qi in train_idx:
            q = all_queries[qi]
            w = worlds[q["world_idx"]]
            gold = set(q["gold"])
            feats = compute_features(w["nodes"], q["text"], w["embs"], w["sim_mat"], embed_model, w["goal"])
            labels = np.array([1.0 if n["id"] in gold else 0.0 for n in w["nodes"]])
            X_lin_parts.append(feats)
            y_lin_parts.append(labels)

        X_lin = np.vstack(X_lin_parts)
        y_lin = np.concatenate(y_lin_parts)
        if len(np.unique(y_lin)) >= 2:
            clf2 = LogisticRegression(max_iter=2000, C=1.0)
            clf2.fit(X_lin, y_lin)
            theta_l = clf2.coef_[0]
        else:
            theta_l = np.ones(7) * 0.5

        # Evaluate test
        for qi in test_idx:
            q = all_queries[qi]
            w = worlds[q["world_idx"]]
            gold = set(q["gold"])
            if not gold:
                continue
            feats = compute_features(w["nodes"], q["text"], w["embs"], w["sim_mat"], embed_model, w["goal"])
            nids = [n["id"] for n in w["nodes"]]
            for method in METHODS:
                scores = retrieve(feats, method, theta_g, theta_l)
                top_idx = np.argsort(scores)[::-1][:K]
                retrieved = [nids[i] for i in top_idx]
                ret_set = set(retrieved)
                recall = len(ret_set & gold) / max(len(gold), 1)
                noise_k = len(ret_set & w["noise"]) / K
                stale_k = len(ret_set & w["stale"]) / K
                dcg = sum(1 / np.log2(i + 2) for i, nid in enumerate(retrieved) if nid in gold)
                idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gold), K)))
                ndcg = dcg / max(idcg, 1e-8)
                results.append({
                    "method": method,
                    "family": q.get("fam", "general"),
                    "recall": recall,
                    "ndcg": ndcg,
                    "noise": noise_k,
                    "stale": stale_k,
                    "fold": fold_i,
                })

        if (fold_i + 1) % 1 == 0:
            print(f"  Fold {fold_i + 1}/{n_folds} done ({len(test_idx)} test queries)")

    df = pd.DataFrame(results)

    # ====================================================================
    # REPORT
    # ====================================================================
    print(f"\n{'=' * 70}")
    print(f"  MGA on LoCoMo-MC10 BENCHMARK — RESULTS")
    print(f"  {len(worlds)} conversations, {len(all_queries)} queries, {n_folds}-fold CV")
    print(f"{'=' * 70}")

    print(f"\n  {'Method':<12} {'Recall@5':>10} {'nDCG@5':>10} {'Noise@5':>10}")
    print(f"  {'-' * 44}")
    for m in METHODS:
        md = df[df["method"] == m]
        print(f"  {m:<12} {md['recall'].mean():>10.4f} {md['ndcg'].mean():>10.4f} {md['noise'].mean():>10.4f}")

    # Per family
    print(f"\n  PER-FAMILY (Recall@5):")
    fams = sorted(df["family"].unique())
    print(f"  {'Family':<15}", end="")
    for m in METHODS:
        print(f" {m:>10}", end="")
    print()
    print(f"  {'-' * 67}")
    for fam in fams:
        fd = df[df["family"] == fam]
        if len(fd) < 5:
            continue
        print(f"  {fam:<15}", end="")
        for m in METHODS:
            r = fd[fd["method"] == m]["recall"]
            print(f" {r.mean():>10.4f}" if len(r) > 0 else f" {'N/A':>10}", end="")
        print()

    # Statistical test: MGA_linear vs GA
    ga_vals = df[df["method"] == "ga"]["recall"].values
    mga_gate_vals = df[df["method"] == "mga_gate"]["recall"].values
    mga_lin_vals = df[df["method"] == "mga_linear"]["recall"].values

    print(f"\n  STATISTICAL TESTS:")
    for name, vals in [("mga_gate", mga_gate_vals), ("mga_linear", mga_lin_vals)]:
        min_len = min(len(vals), len(ga_vals))
        if min_len > 10:
            t_stat, p_val = stats.ttest_rel(vals[:min_len], ga_vals[:min_len])
            diff = vals[:min_len].mean() - ga_vals[:min_len].mean()
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {name} vs GA: diff={diff:+.4f}, t={t_stat:.3f}, p={p_val:.4f} {sig}")

    # Theta
    print(f"\n  LEARNED THETA (LoCoMo, mean across {n_folds} folds):")
    feat_names = ["recency", "frequency", "unresolved", "utility", "weight", "goal_rel"]
    mean_t = np.mean(thetas, axis=0)
    std_t = np.std(thetas, axis=0)
    for name, val, s in zip(feat_names, mean_t, std_t):
        print(f"    {name:<15}: {val:>8.4f} +/- {s:.4f}")

    # Bootstrap CI for headline numbers
    print(f"\n  BOOTSTRAP 95% CI (Recall@5):")
    for m in METHODS:
        vals = df[df["method"] == m]["recall"].values
        boots = [np.mean(np.random.choice(vals, size=len(vals), replace=True)) for _ in range(1000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"    {m:<12}: {vals.mean():.4f} [{lo:.4f}, {hi:.4f}]")

    print(f"\n{'=' * 70}")
    print("  DONE — LoCoMo-MC10 benchmark complete")
    print(f"{'=' * 70}")

    return df.to_dict()


@app.local_entrypoint()
def main():
    result = run_locomo_benchmark.remote()
    print("\nLoCoMo benchmark complete.")
