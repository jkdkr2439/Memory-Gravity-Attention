"""
MGA on LoCoMo Benchmark (Snap Research, 2024)
"Evaluating Very Long-Term Conversational Memory of LLM Agents"

LoCoMo: 300 turns, 35 sessions, 9K tokens avg per conversation.
Has gold QA labels for long-term memory evaluation.

Run: modal run scripts/modal_locomo_benchmark.py
"""

import modal

app = modal.App("mga-locomo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "scikit-learn", "sentence-transformers", "torch", "scipy", "datasets")
)


@app.function(image=image, gpu="T4", timeout=900)
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

    print("Loading model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Loading LoCoMo dataset...")
    try:
        ds = load_dataset("pgmenon/soul-benchmarks-locomo", split="train")
        print(f"Loaded LoCoMo: {len(ds)} items")
    except Exception as e:
        print(f"pgmenon failed ({e}), trying Percena...")
        try:
            ds = load_dataset("Percena/locomo-mc10", split="train")
            print(f"Loaded locomo-mc10: {len(ds)} items")
        except Exception as e2:
            print(f"Both failed. Trying snap-research GitHub...")
            # Fallback: load from the official repo
            import urllib.request, json
            url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo_qa.json"
            resp = urllib.request.urlopen(url)
            ds = json.loads(resp.read().decode())
            print(f"Loaded from GitHub: {len(ds)} items")

    # ====================================================================
    # PROCESS LoCoMo INTO MGA FORMAT
    # ====================================================================
    print("\nProcessing LoCoMo data...")
    
    # Inspect structure
    if hasattr(ds, 'column_names'):
        print(f"Columns: {ds.column_names}")
        sample = ds[0]
    elif isinstance(ds, list):
        sample = ds[0]
    else:
        sample = ds[0]
    
    print(f"Sample keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
    
    # Process based on format
    worlds = []
    
    def process_locomo_item(item, idx):
        """Convert LoCoMo item to MGA world.
        Format: config, conversation, question_idx, question, ground_truth, predicted, category, score
        """
        conversation = item.get("conversation", "")
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        category = item.get("category", "general")
        
        if not conversation or not question:
            return None
        
        # Parse conversation into nodes
        nodes = []
        
        if isinstance(conversation, str):
            # Split by newlines or speaker markers
            lines = [l.strip() for l in conversation.split('\n') if l.strip() and len(l.strip()) > 5]
            if len(lines) < 5:
                # Try splitting by common patterns
                import re
                lines = re.split(r'(?:Speaker \d+:|User:|Assistant:|Human:|AI:|Person \d+:|\[.+?\]:)', conversation)
                lines = [l.strip() for l in lines if l.strip() and len(l.strip()) > 5]
            
            for i, line in enumerate(lines):
                if len(line) < 5:
                    continue
                session = i // 8
                ts = (datetime(2024,1,1,9,0,0) + timedelta(days=session, minutes=i*3)).isoformat()
                
                line_lower = line.lower()
                if any(w in line_lower for w in ["remember", "always", "rule", "must", "note", "important"]):
                    ntype = "state_fact"
                elif any(w in line_lower for w in ["todo", "need to", "still", "pending", "haven't"]):
                    ntype = "open_task"
                elif any(w in line_lower for w in ["actually", "change", "instead", "no wait", "correction"]):
                    ntype = "revision"
                elif len(line) < 25 or any(w in line_lower for w in ["haha", "lol", "ok", "sure", "yeah", "thanks", "bye"]):
                    ntype = "chatter"
                else:
                    ntype = "content"
                
                nodes.append({
                    "id": f"n{len(nodes):04d}",
                    "content": line[:500],
                    "ts": ts,
                    "sess": session,
                    "type": ntype,
                })
        elif isinstance(conversation, list):
            for i, msg in enumerate(conversation):
                content = msg.get("text", msg.get("content", msg.get("utterance", str(msg)))) if isinstance(msg, dict) else str(msg)
                if not content or len(content) < 5:
                    continue
                session = i // 8
                ts = (datetime(2024,1,1,9,0,0) + timedelta(days=session, minutes=i*3)).isoformat()
                nodes.append({
                    "id": f"n{len(nodes):04d}",
                    "content": content[:500],
                    "ts": ts,
                    "sess": session,
                    "type": "content",
                })
        
        if len(nodes) < 8:
            return None
        
        # Gold nodes: find nodes whose content overlaps with ground_truth
        gold_nodes = []
        if ground_truth:
            gt_lower = str(ground_truth).lower()
            gt_words = [w for w in gt_lower.split() if len(w) > 4][:8]
            for n in nodes:
                n_lower = n["content"].lower()
                matches = sum(1 for w in gt_words if w in n_lower)
                if matches >= max(2, len(gt_words) // 3):
                    gold_nodes.append(n["id"])
            gold_nodes = gold_nodes[:5]
        
        if not gold_nodes:
            # Fallback: use last content nodes as loose gold
            content_nodes = [n for n in nodes if n["type"] == "content" and len(n["content"]) > 30]
            if content_nodes:
                gold_nodes = [content_nodes[-1]["id"]]
            else:
                return None
        
        noise = set(n["id"] for n in nodes if n["type"] == "chatter")
        stale = set()
        
        query = {
            "id": f"q_{idx}",
            "fam": str(category) if category else "general",
            "text": str(question),
            "gold": gold_nodes,
        }
        
        return {"nodes": nodes, "queries": [query], "stale": stale, "noise": noise, "n_sessions": nodes[-1]["sess"]+1}
    
    # Process items
    max_items = min(len(ds) if not isinstance(ds, list) else len(ds), 500)
    for i in range(max_items):
        try:
            item = ds[i] if not isinstance(ds, list) else ds[i]
            world = process_locomo_item(item, i)
            if world:
                worlds.append(world)
                if len(worlds) <= 5 or len(worlds) % 20 == 0:
                    print(f"  Processed {len(worlds)} valid items ({world['nodes'].__len__()} nodes)")
        except Exception as e:
            continue
        if len(worlds) >= 100:
            break
    
    print(f"\nTotal valid items: {len(worlds)}")
    
    if len(worlds) < 10:
        print("ERROR: Not enough valid items from LoCoMo.")
        print(f"Processed {max_items} items, got {len(worlds)} valid.")
        if len(worlds) > 0:
            print(f"Sample world: {len(worlds[0]['nodes'])} nodes, query: {worlds[0]['queries'][0]['text'][:60]}")
        return {"error": "insufficient valid data", "n_worlds": len(worlds)}
    
    # ====================================================================
    # EMBED
    # ====================================================================
    print("Embedding...")
    for w in worlds:
        texts = [n["content"] for n in w["nodes"]]
        w["embs"] = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        w["sim_mat"] = cosine_similarity(w["embs"])
        w["goal"] = " ".join(n["content"] for n in w["nodes"] if n["type"] in ("state_fact","content"))[:500]
    
    # ====================================================================
    # FEATURES & RETRIEVERS
    # ====================================================================
    def compute_features(nodes, query_text, embeddings, sim_matrix, embed_model, goal_text):
        n = len(nodes)
        q_emb = embed_model.encode([query_text], normalize_embeddings=True)
        sim_scores = cosine_similarity(q_emb, embeddings)[0]
        t_q = datetime.fromisoformat(nodes[-1]["ts"])
        recency = np.array([np.exp(-(t_q-datetime.fromisoformat(nd["ts"])).total_seconds()/(5*86400)) for nd in nodes])
        freq_counts = np.array([(sim_matrix[i]>0.75).sum()-1 for i in range(n)], dtype=float)
        freq = np.log1p(freq_counts)/max(np.log1p(freq_counts.max()),1e-8)
        task_cues = ["need to","still","todo","pending","waiting"]
        unresolved = np.zeros(n)
        for i,nd in enumerate(nodes):
            if any(c in nd["content"].lower() for c in task_cues):
                unresolved[i] = 1.0
        g_emb = embed_model.encode([goal_text], normalize_embeddings=True)
        goal_rel = cosine_similarity(g_emb, embeddings)[0]
        goal_rel = (goal_rel-goal_rel.min())/max(goal_rel.max()-goal_rel.min(),1e-8)
        sim_norm = (sim_scores-sim_scores.min())/max(sim_scores.max()-sim_scores.min(),1e-8)
        return np.column_stack([sim_norm, recency, freq, unresolved, np.full(n,.5), np.full(n,.5), goal_rel])

    def retrieve(features, method, theta_g=None, theta_l=None):
        sim = features[:,0]
        if method=="recency": return features[:,1]
        elif method=="sim": return sim
        elif method=="ga": return sim + features[:,1] + features[:,5]
        elif method=="mga_gate": return sim*(1.0+expit(features[:,1:]@theta_g))
        elif method=="mga_linear": return features@theta_l
        return sim

    # ====================================================================
    # K-FOLD
    # ====================================================================
    METHODS = ["recency","sim","ga","mga_gate","mga_linear"]
    K = 5

    all_queries = []
    for wi,w in enumerate(worlds):
        for q in w["queries"]:
            all_queries.append({"world_idx":wi, **q})
    
    print(f"Total queries for benchmark: {len(all_queries)}")
    print("Running 5-fold CV...")
    
    kf = KFold(n_splits=min(5, len(all_queries)), shuffle=True, random_state=42)
    results = []
    thetas = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(all_queries)))):
        X_tr, y_tr = [], []
        for qi in train_idx:
            q = all_queries[qi]; w = worlds[q["world_idx"]]
            gold = set(q["gold"])
            feats = compute_features(w["nodes"], q["text"], w["embs"], w["sim_mat"], embed_model, w["goal"])
            labels = np.array([1.0 if n["id"] in gold else 0.0 for n in w["nodes"]])
            X_tr.append(feats[:,1:]); y_tr.append(labels)
        
        X_tr = np.vstack(X_tr); y_tr = np.concatenate(y_tr)
        if len(np.unique(y_tr))>=2:
            clf = LogisticRegression(max_iter=2000,C=1.0); clf.fit(X_tr,y_tr)
            theta_g = clf.coef_[0]
        else:
            theta_g = np.ones(6)*0.5
        thetas.append(theta_g)

        X_lin = np.vstack([compute_features(worlds[all_queries[qi]["world_idx"]]["nodes"],all_queries[qi]["text"],worlds[all_queries[qi]["world_idx"]]["embs"],worlds[all_queries[qi]["world_idx"]]["sim_mat"],embed_model,worlds[all_queries[qi]["world_idx"]]["goal"]) for qi in train_idx])
        y_lin = np.concatenate([np.array([1.0 if n["id"] in set(all_queries[qi]["gold"]) else 0.0 for n in worlds[all_queries[qi]["world_idx"]]["nodes"]]) for qi in train_idx])
        if len(np.unique(y_lin))>=2:
            clf2 = LogisticRegression(max_iter=2000,C=1.0); clf2.fit(X_lin,y_lin)
            theta_l = clf2.coef_[0]
        else:
            theta_l = np.ones(7)*0.5

        for qi in test_idx:
            q = all_queries[qi]; w = worlds[q["world_idx"]]
            gold = set(q["gold"])
            if not gold: continue
            feats = compute_features(w["nodes"],q["text"],w["embs"],w["sim_mat"],embed_model,w["goal"])
            nids = [n["id"] for n in w["nodes"]]
            for method in METHODS:
                scores = retrieve(feats,method,theta_g,theta_l)
                top_idx = np.argsort(scores)[::-1][:K]
                retrieved = [nids[i] for i in top_idx]
                ret_set = set(retrieved)
                recall = len(ret_set&gold)/max(len(gold),1)
                noise_k = len(ret_set&w["noise"])/K
                stale_k = len(ret_set&w["stale"])/K
                dcg = sum(1/np.log2(i+2) for i,nid in enumerate(retrieved) if nid in gold)
                idcg = sum(1/np.log2(i+2) for i in range(min(len(gold),K)))
                ndcg = dcg/max(idcg,1e-8)
                results.append({"method":method,"family":q.get("fam","general"),"recall":recall,"ndcg":ndcg,"noise":noise_k,"stale":stale_k,"fold":fold_i})

    df = pd.DataFrame(results)

    # ====================================================================
    # REPORT
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"  MGA on LoCoMo BENCHMARK")
    print(f"  {len(worlds)} conversations, {len(all_queries)} queries, 5-fold CV")
    print(f"{'='*70}")

    print(f"\n  {'Method':<12} {'Recall@5':>10} {'nDCG@5':>10} {'Noise@5':>10}")
    print(f"  {'-'*44}")
    for m in METHODS:
        md = df[df["method"]==m]
        print(f"  {m:<12} {md['recall'].mean():>10.4f} {md['ndcg'].mean():>10.4f} {md['noise'].mean():>10.4f}")

    # Per family
    print(f"\n  PER-FAMILY (Recall@5):")
    fams = sorted(df["family"].unique())[:10]
    for fam in fams:
        fd = df[df["family"]==fam]
        if len(fd) < 5: continue
        print(f"  {str(fam)[:20]:<20}", end="")
        for m in METHODS:
            r = fd[fd["method"]==m]["recall"]
            print(f" {r.mean():.4f}" if len(r)>0 else " N/A  ", end="")
        print()

    # Statistical test
    ga_vals = df[df["method"]=="ga"]["recall"].values
    mga_vals = df[df["method"]=="mga_linear"]["recall"].values
    min_len = min(len(ga_vals), len(mga_vals))
    if min_len > 5:
        t_stat, p_val = stats.ttest_rel(mga_vals[:min_len], ga_vals[:min_len])
        diff = mga_vals[:min_len].mean() - ga_vals[:min_len].mean()
        print(f"\n  STATISTICAL TEST (MGA_linear vs GA):")
        print(f"  Paired t-test: t={t_stat:.4f}, p={p_val:.4f} {'***' if p_val<0.001 else '**' if p_val<0.01 else '*' if p_val<0.05 else 'ns'}")
        print(f"  Mean diff: {diff:+.4f}")

    # Theta
    print(f"\n  LEARNED THETA (LoCoMo):")
    feat_names = ["recency","frequency","unresolved","utility","weight","goal_rel"]
    mean_t = np.mean(thetas, axis=0)
    for name,val in zip(feat_names, mean_t):
        print(f"    {name:<15}: {val:>8.4f}")

    print(f"\n{'='*70}")
    print("  DONE - LoCoMo benchmark complete")
    print(f"{'='*70}")
    return df.to_dict()


@app.local_entrypoint()
def main():
    result = run_locomo_benchmark.remote()
    print("\nLoCoMo benchmark complete.")
