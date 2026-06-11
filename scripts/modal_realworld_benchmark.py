"""
MGA Real-World Benchmark on Modal.com
Uses ShareGPT conversations as real multi-turn dialogue data.

Pipeline:
1. Load ShareGPT conversations from HuggingFace
2. Filter long conversations (>15 turns)
3. Split into sessions (every 6 turns = 1 session)
4. Create memory nodes from assistant/user messages
5. Generate queries (retrieve specific facts mentioned earlier)
6. Run MGA vs baselines with K-fold CV

Run: modal run scripts/modal_realworld_benchmark.py
"""

import modal

app = modal.App("mga-realworld")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "scikit-learn", "sentence-transformers", "torch", "scipy", "datasets")
)


@app.function(image=image, gpu="T4", timeout=900)
def run_realworld_benchmark():
    import numpy as np
    import pandas as pd
    import random
    from datetime import datetime, timedelta
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from scipy.special import expit
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    print("Loading sentence-transformers...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Loading ShareGPT dataset...")
    
    # Load a public multi-turn conversation dataset
    try:
        ds = load_dataset("OpenAssistant/oasst2", split="train")
        print(f"Loaded OASST2: {len(ds)} messages")
        # OASST2 is a tree structure - group by conversation
        # Group messages by tree_id (conversation_id)
        from collections import defaultdict
        convos = defaultdict(list)
        for item in ds:
            tree_id = item.get("message_tree_id", item.get("parent_id", "unknown"))
            convos[tree_id].append(item)
        # Convert to list of conversations
        conv_list = []
        for tree_id, msgs in convos.items():
            # Sort by created_date or rank
            msgs.sort(key=lambda x: x.get("created_date", "") or "")
            conv_list.append({"conversations": [{"value": m.get("text", ""), "from": m.get("role", "user")} for m in msgs]})
        ds = conv_list
        print(f"Grouped into {len(ds)} conversations")
    except Exception as e:
        print(f"OASST2 failed ({e}), trying UltraChat...")
        ds = load_dataset("stingning/ultrachat", split="train")
        print(f"Loaded UltraChat: {len(ds)} conversations")
        # UltraChat format: list of messages per item
        conv_list = []
        for item in ds:
            data = item.get("data", item.get("messages", []))
            if isinstance(data, list) and len(data) > 0:
                conv_list.append({"conversations": [{"value": m if isinstance(m, str) else m.get("content",""), "from": "user" if i%2==0 else "assistant"} for i, m in enumerate(data)]})
        ds = conv_list
        print(f"Converted {len(ds)} conversations")
    
    # ====================================================================
    # PROCESS CONVERSATIONS INTO MEMORY WORLDS
    # ====================================================================
    
    def process_conversation(conv_data, conv_idx, min_turns=15, session_size=6):
        """Convert a conversation into MGA world format."""
        
        # Extract messages
        if "conversations" in conv_data:
            messages = conv_data["conversations"]
        elif "conversation" in conv_data:
            messages = conv_data["conversation"]
        else:
            return None
        
        if not messages or len(messages) < min_turns:
            return None
        
        # Build nodes from messages
        nodes = []
        base_time = datetime(2024, 1, 1, 9, 0, 0) + timedelta(hours=conv_idx)
        
        for i, msg in enumerate(messages):
            # Get content
            if isinstance(msg, dict):
                content = msg.get("value", msg.get("content", ""))
                role = msg.get("from", msg.get("role", "user"))
            elif isinstance(msg, str):
                content = msg
                role = "user" if i % 2 == 0 else "assistant"
            else:
                continue
            
            if not content or len(content) < 10:
                continue
            
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500]
            
            session_id = i // session_size
            ts = base_time + timedelta(minutes=i * 5)
            
            # Classify message type
            content_lower = content.lower()
            if any(w in content_lower for w in ["remember", "always", "rule", "must", "important", "note that"]):
                msg_type = "state_fact"
            elif any(w in content_lower for w in ["todo", "need to", "still", "pending", "haven't", "not yet"]):
                msg_type = "open_task"
            elif any(w in content_lower for w in ["done", "finished", "completed", "fixed", "resolved"]):
                msg_type = "close_task"
            elif any(w in content_lower for w in ["actually", "change", "instead", "update", "correction", "no wait"]):
                msg_type = "revision"
            elif len(content) < 30 or any(w in content_lower for w in ["haha", "lol", "thanks", "ok", "sure", "got it"]):
                msg_type = "chatter"
            else:
                msg_type = "content"
            
            nodes.append({
                "id": f"n{len(nodes):04d}",
                "content": content,
                "ts": ts.isoformat(),
                "sess": session_id,
                "type": msg_type,
                "role": role,
            })
        
        if len(nodes) < 15:
            return None
        
        # Oracle: classify stale/noise
        noise = set(n["id"] for n in nodes if n["type"] == "chatter")
        stale = set()  # In real data, hard to determine without ground truth
        
        # For real data, we use a proxy: if same topic mentioned later with "actually/change/instead",
        # earlier mentions on that topic become "stale"
        revision_nodes = [n for n in nodes if n["type"] == "revision"]
        for rev in revision_nodes:
            # Mark earlier nodes with high similarity to revision as potentially stale
            # (will be computed after embedding)
            pass
        
        # Generate queries from the conversation itself
        queries = []
        
        # Query type 1: "What was said about X?" (retrieve specific earlier content)
        # Use nodes from first half as gold, ask about them
        mid = len(nodes) // 2
        early_content_nodes = [n for n in nodes[:mid] if n["type"] in ("state_fact", "content") and len(n["content"]) > 50]
        
        if early_content_nodes:
            # Pick up to 3 early nodes as needle queries
            for needle in early_content_nodes[:3]:
                # Create query from first 50 chars
                query_text = f"What was mentioned earlier about: {needle['content'][:60]}?"
                queries.append({
                    "id": f"q_needle_{needle['id']}",
                    "fam": "needle",
                    "text": query_text,
                    "gold": [needle["id"]],
                })
        
        # Query type 2: "What tasks are pending?" (open-loop)
        open_tasks = [n for n in nodes if n["type"] == "open_task"]
        closed_tasks = [n for n in nodes if n["type"] == "close_task"]
        # Simple heuristic: open tasks not followed by close
        if open_tasks:
            queries.append({
                "id": "q_open",
                "fam": "open_loop",
                "text": "What tasks or items are still pending or unresolved?",
                "gold": [n["id"] for n in open_tasks[-3:]],  # most recent open tasks
            })
        
        # Query type 3: "What are the key facts/rules?" (constraint recall)
        fact_nodes = [n for n in nodes if n["type"] == "state_fact"]
        if fact_nodes:
            queries.append({
                "id": "q_facts",
                "fam": "constraint",
                "text": "What are the key rules, facts, or constraints mentioned?",
                "gold": [n["id"] for n in fact_nodes[-5:]],
            })
        
        # Query type 4: "What changed?" (change detection)
        if revision_nodes:
            queries.append({
                "id": "q_change",
                "fam": "change_detect",
                "text": "What was changed, corrected, or updated during the conversation?",
                "gold": [n["id"] for n in revision_nodes],
            })
        
        # Query type 5: General noise resistance
        non_noise = [n for n in nodes if n["type"] not in ("chatter",) and len(n["content"]) > 50]
        if non_noise and len(noise) > 3:
            queries.append({
                "id": "q_noise",
                "fam": "noise",
                "text": "What are the substantive points discussed (not small talk)?",
                "gold": [n["id"] for n in non_noise[-8:]],
            })
        
        if len(queries) < 3:
            return None
        
        return {"nodes": nodes, "queries": queries, "stale": stale, "noise": noise, "n_sessions": nodes[-1]["sess"] + 1}
    
    # ====================================================================
    # PROCESS DATASET
    # ====================================================================
    print("\nProcessing conversations into MGA worlds...")
    worlds = []
    max_worlds = 20
    
    for i in range(min(len(ds), 500)):  # scan up to 500 conversations
        if len(worlds) >= max_worlds:
            break
        try:
            world = process_conversation(ds[i], i)
            if world and len(world["queries"]) >= 3:
                worlds.append(world)
                print(f"  World {len(worlds)-1}: {len(world['nodes'])} nodes, {len(world['queries'])} queries, {world['n_sessions']} sessions")
        except Exception as e:
            continue
    
    if len(worlds) < 5:
        print(f"WARNING: Only {len(worlds)} valid worlds. Need at least 5.")
        print("Trying with lower min_turns...")
        for i in range(500, min(len(ds), 2000)):
            if len(worlds) >= max_worlds:
                break
            try:
                world = process_conversation(ds[i], i, min_turns=10, session_size=4)
                if world and len(world["queries"]) >= 3:
                    worlds.append(world)
                    print(f"  World {len(worlds)-1}: {len(world['nodes'])} nodes, {len(world['queries'])} queries")
            except:
                continue
    
    print(f"\nTotal worlds: {len(worlds)}")
    if len(worlds) < 3:
        print("ERROR: Not enough valid conversations. Aborting.")
        return {"error": "insufficient data"}
    
    # ====================================================================
    # EMBED & FEATURES
    # ====================================================================
    print("\nEmbedding all nodes...")
    for wi, w in enumerate(worlds):
        texts = [n["content"] for n in w["nodes"]]
        w["embs"] = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        w["sim_mat"] = cosine_similarity(w["embs"])
        # Mark stale based on revision similarity
        for rev_node in [n for n in w["nodes"] if n["type"] == "revision"]:
            rev_idx = next(i for i, n in enumerate(w["nodes"]) if n["id"] == rev_node["id"])
            for j in range(rev_idx):
                if w["sim_mat"][rev_idx, j] > 0.6 and w["nodes"][j]["type"] in ("state_fact", "content"):
                    w["stale"].add(w["nodes"][j]["id"])
        # Goal text
        w["goal"] = " ".join(n["content"] for n in w["nodes"] if n["type"] in ("state_fact", "content") and n["id"] not in w["stale"])[:1000]
    
    # ====================================================================
    # FEATURES & RETRIEVERS (same as before)
    # ====================================================================
    def compute_features(nodes, query_text, embeddings, sim_matrix, embed_model, goal_text):
        n = len(nodes)
        q_emb = embed_model.encode([query_text], normalize_embeddings=True)
        sim_scores = cosine_similarity(q_emb, embeddings)[0]
        t_q = datetime.fromisoformat(nodes[-1]["ts"])
        recency = np.array([np.exp(-(t_q - datetime.fromisoformat(nd["ts"])).total_seconds() / (5*86400)) for nd in nodes])
        freq_counts = np.array([(sim_matrix[i] > 0.75).sum() - 1 for i in range(n)], dtype=float)
        freq = np.log1p(freq_counts) / max(np.log1p(freq_counts.max()), 1e-8)
        task_cues = ["need to", "still", "todo", "pending", "waiting", "haven't", "not yet"]
        closure_cues = ["done", "fixed", "completed", "finished", "resolved"]
        unresolved = np.zeros(n)
        for i, nd in enumerate(nodes):
            cl = nd["content"].lower()
            if any(c in cl for c in task_cues):
                has_close = any(any(c in nodes[j]["content"].lower() for c in closure_cues) and sim_matrix[i,j] > 0.4 for j in range(i+1, n))
                unresolved[i] = 0.0 if has_close else 1.0
        g_emb = embed_model.encode([goal_text], normalize_embeddings=True)
        goal_rel = cosine_similarity(g_emb, embeddings)[0]
        goal_rel = (goal_rel - goal_rel.min()) / max(goal_rel.max() - goal_rel.min(), 1e-8)
        sim_norm = (sim_scores - sim_scores.min()) / max(sim_scores.max() - sim_scores.min(), 1e-8)
        return np.column_stack([sim_norm, recency, freq, unresolved, np.full(n, .5), np.full(n, .5), goal_rel])

    def retrieve(features, method, theta_g=None, theta_l=None):
        sim = features[:, 0]
        if method == "recency": return features[:, 1]
        elif method == "sim": return sim
        elif method == "ga": return sim + features[:, 1] + features[:, 5]
        elif method == "mga_gate": return sim * (1.0 + expit(features[:, 1:] @ theta_g))
        elif method == "mga_linear": return features @ theta_l
        return sim

    # ====================================================================
    # K-FOLD BENCHMARK
    # ====================================================================
    METHODS = ["recency", "sim", "ga", "mga_gate", "mga_linear"]
    K = 5

    # Pool queries
    all_queries = []
    for wi, w in enumerate(worlds):
        for q in w["queries"]:
            all_queries.append({"world_idx": wi, **q})
    
    print(f"\nTotal queries: {len(all_queries)}")
    print(f"Running 5-fold CV...")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    thetas_g = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(all_queries)))):
        # Learn theta
        X_tr, y_tr = [], []
        for qi in train_idx:
            q = all_queries[qi]
            w = worlds[q["world_idx"]]
            gold = set(q["gold"])
            feats = compute_features(w["nodes"], q["text"], w["embs"], w["sim_mat"], embed_model, w["goal"])
            labels = np.array([1.0 if n["id"] in gold else 0.0 for n in w["nodes"]])
            X_tr.append(feats[:, 1:]); y_tr.append(labels)
        
        X_tr = np.vstack(X_tr); y_tr = np.concatenate(y_tr)
        if len(np.unique(y_tr)) >= 2:
            clf = LogisticRegression(max_iter=2000, C=1.0); clf.fit(X_tr, y_tr)
            theta_g = clf.coef_[0]
        else:
            theta_g = np.ones(6) * 0.5
        thetas_g.append(theta_g)

        # Linear
        X_lin = np.vstack([compute_features(worlds[all_queries[qi]["world_idx"]]["nodes"], all_queries[qi]["text"], worlds[all_queries[qi]["world_idx"]]["embs"], worlds[all_queries[qi]["world_idx"]]["sim_mat"], embed_model, worlds[all_queries[qi]["world_idx"]]["goal"]) for qi in train_idx])
        y_lin = np.concatenate([np.array([1.0 if n["id"] in set(all_queries[qi]["gold"]) else 0.0 for n in worlds[all_queries[qi]["world_idx"]]["nodes"]]) for qi in train_idx])
        if len(np.unique(y_lin)) >= 2:
            clf2 = LogisticRegression(max_iter=2000, C=1.0); clf2.fit(X_lin, y_lin)
            theta_l = clf2.coef_[0]
        else:
            theta_l = np.ones(7) * 0.5

        # Eval
        for qi in test_idx:
            q = all_queries[qi]
            w = worlds[q["world_idx"]]
            gold = set(q["gold"])
            if not gold: continue
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
                dcg = sum(1/np.log2(i+2) for i, nid in enumerate(retrieved) if nid in gold)
                idcg = sum(1/np.log2(i+2) for i in range(min(len(gold), K)))
                ndcg = dcg / max(idcg, 1e-8)
                results.append({"method": method, "family": q.get("fam", "?"), "recall": recall, "ndcg": ndcg, "noise": noise_k, "stale": stale_k, "fold": fold_i})

    df = pd.DataFrame(results)

    # ====================================================================
    # REPORT
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"  MGA REAL-WORLD BENCHMARK (ShareGPT data)")
    print(f"  {len(worlds)} conversations, {len(all_queries)} queries, 5-fold CV")
    print(f"{'='*70}")

    print(f"\n  {'Method':<12} {'Recall@5':>10} {'nDCG@5':>10} {'Noise@5':>10} {'Stale@5':>10}")
    print(f"  {'-'*54}")
    for m in METHODS:
        md = df[df["method"] == m]
        print(f"  {m:<12} {md['recall'].mean():>10.4f} {md['ndcg'].mean():>10.4f} {md['noise'].mean():>10.4f} {md['stale'].mean():>10.4f}")

    print(f"\n  PER-FAMILY (Recall@5):")
    fams = sorted(df["family"].unique())
    print(f"  {'Family':<15}", end="")
    for m in METHODS: print(f" {m:>10}", end="")
    print()
    for fam in fams:
        fd = df[df["family"] == fam]
        print(f"  {fam:<15}", end="")
        for m in METHODS:
            r = fd[fd["method"] == m]["recall"]
            print(f" {r.mean():>10.4f}" if len(r) > 0 else f" {'N/A':>10}", end="")
        print()

    # Statistical test
    print(f"\n  STATISTICAL TEST (MGA_linear vs GA):")
    ga_r = []; mga_r = []
    for qi in range(len(all_queries)):
        g = df[(df["method"] == "ga")].iloc[qi % len(df[df["method"]=="ga"])]["recall"] if qi < len(df[df["method"]=="ga"]) else 0
        m = df[(df["method"] == "mga_linear")].iloc[qi % len(df[df["method"]=="mga_linear"])]["recall"] if qi < len(df[df["method"]=="mga_linear"]) else 0
    
    ga_vals = df[df["method"]=="ga"]["recall"].values
    mga_vals = df[df["method"]=="mga_linear"]["recall"].values
    min_len = min(len(ga_vals), len(mga_vals))
    diff = mga_vals[:min_len] - ga_vals[:min_len]
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(mga_vals[:min_len], ga_vals[:min_len])
    print(f"  Paired t-test: t={t_stat:.4f}, p={p_val:.4f} {'*' if p_val < 0.05 else 'ns'}")
    print(f"  Mean diff (MGA - GA): {diff.mean():+.4f}")

    # Theta
    print(f"\n  LEARNED THETA (real data):")
    feat_names = ["recency", "frequency", "unresolved", "utility", "weight", "goal_rel"]
    mean_t = np.mean(thetas_g, axis=0)
    for name, val in zip(feat_names, mean_t):
        print(f"    {name:<15}: {val:>8.4f}")

    print(f"\n{'='*70}")
    print("  DONE - Real-world benchmark complete")
    print(f"{'='*70}")
    
    return df.to_dict()


@app.local_entrypoint()
def main():
    result = run_realworld_benchmark.remote()
    print("\nReal-world benchmark complete.")
