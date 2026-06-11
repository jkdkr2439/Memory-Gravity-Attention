"""
MGA Benchmark on Modal.com
Run with: modal run scripts/modal_benchmark.py
"""

import modal

app = modal.App("mga-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas",
        "scikit-learn",
        "sentence-transformers",
        "torch",
    )
)


@app.function(image=image, gpu="T4", timeout=600)
def run_mga_benchmark():
    """Full MGA benchmark with real embeddings on GPU."""
    
    import numpy as np
    import pandas as pd
    import random
    import json
    from datetime import datetime, timedelta
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import LogisticRegression
    from scipy.special import expit
    from sentence_transformers import SentenceTransformer
    
    print("Loading sentence-transformers model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.")
    
    # ========== TEMPLATES ==========
    TEMPLATES = {
        "state_fact": [
            "By the way, {key} is {value}.",
            "Remember: {key} should be {value}.",
            "For this project, {key} = {value}.",
            "Note: always use {value} for {key}.",
            "Important: {key} is set to {value}.",
            "Just to be clear, the {key} we're using is {value}.",
            "FYI: {key} has been set to {value}.",
            "Please keep in mind that {key} is {value}.",
        ],
        "revise_fact": [
            "Actually, change {key} to {value} now.",
            "Update: {key} is now {value} instead.",
            "Correction: switch {key} to {value}.",
            "New rule: {key} should be {value} going forward.",
            "I changed my mind — {key} is now {value}.",
            "Scratch the old {key}. Use {value} from now on.",
        ],
        "open_task": [
            "We still need to {task}.",
            "TODO: {task}.",
            "Pending: {task} is not done yet.",
            "Don't forget — {task} remains open.",
            "Still waiting on: {task}.",
            "This hasn't been addressed: {task}.",
            "Can someone handle {task}? It's still pending.",
            "Reminder: {task} is unresolved.",
        ],
        "close_task": [
            "{task} is done now.",
            "Fixed: {task}.",
            "Completed: {task}.",
            "That's finished — {task} is resolved.",
            "Good news: {task} is taken care of.",
            "Resolved: {task}. Moving on.",
        ],
        "reference": [
            "As mentioned before, {key} is {value}.",
            "Reminder: {key} remains {value}.",
            "Don't forget that {key} = {value}.",
            "Just reiterating: {key} is still {value}.",
            "Per our earlier discussion, {key} is {value}.",
        ],
        "chatter": [
            "Had great coffee today.",
            "The weather is nice outside.",
            "I watched a good movie last night.",
            "Random thought: pineapple on pizza is fine.",
            "Lunch was pretty decent today.",
            "My cat knocked over a glass again.",
            "Traffic was terrible this morning.",
            "I need to buy groceries later.",
            "Anyone else tired today?",
            "Just saw a funny meme.",
            "Weekend plans: probably sleep.",
            "This coffee is really strong.",
            "I should exercise more.",
            "The new season of that show is out.",
            "My neighbor's dog is barking again.",
        ],
    }
    
    DOMAINS = [
        {
            "name": "research_paper",
            "preferences": [
                ("citation_style", ["APA 7", "IEEE", "Chicago", "Harvard", "Vancouver"]),
                ("writing_tone", ["formal", "technical", "accessible", "academic", "concise"]),
                ("language", ["English", "Vietnamese", "bilingual", "French"]),
                ("math_notation", ["LaTeX inline", "display equations", "minimal math", "heavy notation"]),
                ("section_format", ["numbered", "unnumbered", "short sections", "long chapters"]),
                ("figure_style", ["minimalist", "detailed", "colorful", "grayscale"]),
                ("abstract_length", ["150 words", "200 words", "300 words", "one paragraph"]),
            ],
            "tasks": [
                "fix the margin overflow in section 4",
                "add missing references to the bibliography",
                "rewrite the abstract to be more concise",
                "check all equations for consistency",
                "proofread the conclusion",
                "add a figure for the main result",
                "format the appendix properly",
                "resolve the conflicting notation in section 3",
                "update the related work section",
                "fix the table formatting in section 5",
                "add acknowledgments section",
                "check for plagiarism in the introduction",
            ],
        },
        {
            "name": "software_project",
            "preferences": [
                ("framework", ["PyTorch", "TensorFlow", "JAX", "NumPy only", "Keras"]),
                ("testing", ["pytest", "unittest", "no tests", "property-based", "integration"]),
                ("code_style", ["PEP8", "Google style", "minimal comments", "verbose docs", "type hints"]),
                ("deployment", ["Docker", "bare metal", "cloud functions", "local only", "Kubernetes"]),
                ("version_control", ["git flow", "trunk-based", "feature branches", "monorepo"]),
                ("database", ["PostgreSQL", "MongoDB", "SQLite", "Redis", "none"]),
                ("logging", ["structured JSON", "plain text", "minimal", "verbose debug"]),
            ],
            "tasks": [
                "fix the memory leak in the training loop",
                "add unit tests for the data loader",
                "refactor the config system",
                "optimize the inference speed",
                "update the README with new instructions",
                "fix the broken CI pipeline",
                "add logging to the evaluation script",
                "handle the edge case in batch processing",
                "migrate the database schema",
                "fix the authentication bug",
                "add rate limiting to the API",
                "write documentation for the new feature",
            ],
        },
        {
            "name": "design_project",
            "preferences": [
                ("color_palette", ["warm tones", "cool blues", "monochrome", "vibrant", "pastel"]),
                ("typography", ["sans-serif", "serif", "mixed", "custom font", "system fonts"]),
                ("layout", ["grid-based", "free-form", "responsive first", "mobile first", "desktop focus"]),
                ("animation", ["minimal", "smooth transitions", "playful", "none", "subtle hover"]),
                ("accessibility", ["WCAG AA", "WCAG AAA", "basic", "enhanced", "standard"]),
                ("brand_voice", ["professional", "friendly", "bold", "minimalist", "premium"]),
            ],
            "tasks": [
                "redesign the landing page hero section",
                "fix the mobile navigation bug",
                "create icon set for the dashboard",
                "update the color scheme per new brand guidelines",
                "add dark mode support",
                "fix the form validation UX",
                "design the onboarding flow",
                "create responsive email templates",
                "fix the image loading performance",
                "add micro-interactions to buttons",
            ],
        },
    ]
    
    # ========== GENERATOR ==========
    def generate_world(seed, n_sessions=25, events_per_session=(10, 25)):
        rng = random.Random(seed)
        domain = rng.choice(DOMAINS)
        
        preferences = {}
        for key, options in domain["preferences"]:
            preferences[key] = rng.choice(options)
        
        n_revisions = rng.randint(2, 4)
        revision_keys = rng.sample(list(preferences.keys()), min(n_revisions, len(preferences)))
        revision_schedule = {}
        for key in revision_keys:
            rev_session = rng.randint(n_sessions // 4, n_sessions - 3)
            options = [v for _, opts in domain["preferences"] for v in opts if v != preferences[key]]
            new_val = rng.choice(options) if options else preferences[key] + "_v2"
            revision_schedule[key] = (rev_session, new_val)
        
        all_tasks = domain["tasks"][:]
        rng.shuffle(all_tasks)
        n_tasks = rng.randint(4, min(8, len(all_tasks)))
        open_tasks = all_tasks[:n_tasks]
        task_open_session = {t: rng.randint(0, n_sessions // 3) for t in open_tasks}
        n_close = rng.randint(2, max(2, n_tasks - 2))
        tasks_to_close = rng.sample(open_tasks, n_close)
        task_close_session = {t: rng.randint(task_open_session[t] + 3, n_sessions - 1) for t in tasks_to_close}
        
        nodes = []
        node_id = 0
        current_prefs = dict(preferences)
        current_open_tasks = set()
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        for session in range(n_sessions):
            session_time = base_time + timedelta(days=session, hours=rng.randint(0, 8))
            n_events = rng.randint(*events_per_session)
            
            for key, (rev_s, new_val) in revision_schedule.items():
                if session == rev_s:
                    template = rng.choice(TEMPLATES["revise_fact"])
                    content = template.format(key=key, value=new_val)
                    ts = session_time + timedelta(minutes=rng.randint(1, 30))
                    nodes.append({"id": f"n{node_id:04d}", "content": content, "timestamp": ts.isoformat(), "session_id": session, "type": "revise_fact", "meta_key": key, "meta_val": new_val})
                    current_prefs[key] = new_val
                    node_id += 1
            
            for task, open_s in task_open_session.items():
                if session == open_s:
                    template = rng.choice(TEMPLATES["open_task"])
                    content = template.format(task=task)
                    ts = session_time + timedelta(minutes=rng.randint(5, 40))
                    nodes.append({"id": f"n{node_id:04d}", "content": content, "timestamp": ts.isoformat(), "session_id": session, "type": "open_task", "meta_task": task})
                    current_open_tasks.add(task)
                    node_id += 1
            
            for task in tasks_to_close:
                if task in task_close_session and session == task_close_session[task]:
                    template = rng.choice(TEMPLATES["close_task"])
                    content = template.format(task=task)
                    ts = session_time + timedelta(minutes=rng.randint(10, 50))
                    nodes.append({"id": f"n{node_id:04d}", "content": content, "timestamp": ts.isoformat(), "session_id": session, "type": "close_task", "meta_task": task})
                    current_open_tasks.discard(task)
                    node_id += 1
            
            for i in range(n_events):
                ts = session_time + timedelta(minutes=rng.randint(1, 60) + i * 3)
                etype = rng.choices(["state_fact", "reference", "chatter"], weights=[0.25, 0.30, 0.45], k=1)[0]
                
                if etype in ("state_fact", "reference"):
                    key = rng.choice(list(current_prefs.keys()))
                    template = rng.choice(TEMPLATES[etype])
                    content = template.format(key=key, value=current_prefs[key])
                    nodes.append({"id": f"n{node_id:04d}", "content": content, "timestamp": ts.isoformat(), "session_id": session, "type": etype, "meta_key": key, "meta_val": current_prefs[key]})
                else:
                    content = rng.choice(TEMPLATES["chatter"])
                    nodes.append({"id": f"n{node_id:04d}", "content": content, "timestamp": ts.isoformat(), "session_id": session, "type": "chatter"})
                node_id += 1
        
        # Oracle
        stale_nodes = []
        noise_nodes = [n["id"] for n in nodes if n["type"] == "chatter"]
        for n in nodes:
            if n["type"] in ("state_fact", "reference") and "meta_key" in n:
                if n["meta_val"] != current_prefs.get(n["meta_key"], ""):
                    stale_nodes.append(n["id"])
        
        # Queries
        queries = []
        for key, value in current_prefs.items():
            gold = [n["id"] for n in nodes if n.get("meta_key") == key and n.get("meta_val") == value and n["id"] not in stale_nodes]
            if gold:
                queries.append({"id": f"q_c_{key}", "family": "constraint", "text": f"What is the current {key}?", "gold": gold})
                stale_for_key = [n["id"] for n in nodes if n.get("meta_key") == key and n["id"] in stale_nodes]
                if stale_for_key:
                    queries.append({"id": f"q_s_{key}", "family": "stale_trap", "text": f"What is the latest {key}?", "gold": gold})
        
        if current_open_tasks:
            gold_tasks = [n["id"] for n in nodes if n["type"] == "open_task" and n.get("meta_task") in current_open_tasks]
            if gold_tasks:
                queries.append({"id": "q_open", "family": "open_loop", "text": "What tasks are still open?", "gold": gold_tasks})
        
        # Needle query (fact stated once, long ago)
        early_facts = [n for n in nodes if n["type"] == "state_fact" and n["session_id"] < 3 and n["id"] not in stale_nodes]
        if early_facts:
            needle = rng.choice(early_facts)
            queries.append({"id": "q_needle", "family": "needle", "text": f"What was mentioned about {needle.get('meta_key', 'the early setup')}?", "gold": [needle["id"]]})
        
        # Noise resistance
        gold_facts = [n["id"] for n in nodes if n["type"] in ("state_fact", "revise_fact") and n["id"] not in stale_nodes and n["id"] not in noise_nodes]
        if gold_facts:
            queries.append({"id": "q_noise", "family": "noise", "text": "What are the key project rules and constraints?", "gold": gold_facts[:10]})
        
        rng.shuffle(queries)
        return {"nodes": nodes, "queries": queries, "stale": set(stale_nodes), "noise": set(noise_nodes), "domain": domain["name"], "prefs": current_prefs}
    
    # ========== ESTIMATORS ==========
    def compute_features(nodes, query_text, query_time, embeddings, sim_matrix, embed_model, goal_text):
        n = len(nodes)
        
        # Similarity
        q_emb = embed_model.encode([query_text], normalize_embeddings=True)
        sim_scores = cosine_similarity(q_emb, embeddings)[0]
        
        # Recency
        t_q = datetime.fromisoformat(query_time)
        recency = np.array([np.exp(-(t_q - datetime.fromisoformat(nd["timestamp"])).total_seconds() / (5 * 86400)) for nd in nodes])
        
        # Frequency
        freq_counts = np.array([(sim_matrix[i] > 0.75).sum() - 1 for i in range(n)], dtype=float)
        freq = np.log1p(freq_counts) / max(np.log1p(freq_counts.max()), 1e-8)
        
        # Unresolved
        task_cues = ["need to", "still", "todo", "pending", "waiting", "not done", "remains", "fix", "resolve", "hasn't been"]
        closure_cues = ["done", "fixed", "completed", "finished", "resolved", "cancelled", "taken care"]
        unresolved = np.zeros(n)
        for i, nd in enumerate(nodes):
            cl = nd["content"].lower()
            if any(c in cl for c in task_cues):
                has_close = False
                for j in range(i+1, n):
                    oj = nodes[j]["content"].lower()
                    if any(c in oj for c in closure_cues) and sim_matrix[i,j] > 0.4:
                        has_close = True
                        break
                unresolved[i] = 0.0 if has_close else 1.0
        
        # Goal relevance
        g_emb = embed_model.encode([goal_text], normalize_embeddings=True)
        goal_rel = cosine_similarity(g_emb, embeddings)[0]
        goal_rel = (goal_rel - goal_rel.min()) / max(goal_rel.max() - goal_rel.min(), 1e-8)
        
        # Normalize sim
        sim_norm = (sim_scores - sim_scores.min()) / max(sim_scores.max() - sim_scores.min(), 1e-8)
        
        features = np.column_stack([sim_norm, recency, freq, unresolved, np.full(n, 0.5), np.full(n, 0.5), goal_rel])
        return features, sim_scores
    
    # ========== RETRIEVERS ==========
    def retrieve(features, method, theta=None):
        sim = features[:, 0]
        if method == "recency":
            return features[:, 1]
        elif method == "sim":
            return sim
        elif method == "ga":
            return sim + features[:, 1] + features[:, 5]  # sim + recency + weight
        elif method == "mga_gate":
            f_pers = features[:, 1:]
            gate = expit(f_pers @ theta)
            return sim * (1.0 + gate)
        elif method == "mga_linear":
            return features @ theta
        return sim
    
    # ========== RUN ==========
    print("\nGenerating worlds and running benchmark...")
    
    N_WORLDS = 10
    K = 5
    all_results = []
    all_theta_gate = []
    
    for seed in range(N_WORLDS):
        world = generate_world(seed, n_sessions=25)
        nodes = world["nodes"]
        queries = world["queries"]
        stale_set = world["stale"]
        noise_set = world["noise"]
        
        print(f"  World {seed}: {len(nodes)} nodes, {len(queries)} queries, domain={world['domain']}")
        
        # Embed all nodes
        texts = [n["content"] for n in nodes]
        embeddings = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)
        
        query_time = nodes[-1]["timestamp"]
        goal_text = " ".join(n["content"] for n in nodes if n["type"] in ("state_fact", "revise_fact") and n["id"] not in stale_set)[:1000]
        node_ids = [n["id"] for n in nodes]
        
        # Train/test split
        n_q = len(queries)
        train_end = max(int(n_q * 0.4), 2)
        train_qs = queries[:train_end]
        test_qs = queries[train_end:]
        
        # Learn theta
        X_train, y_train = [], []
        for q in train_qs:
            gold = set(q["gold"])
            feats, _ = compute_features(nodes, q["text"], query_time, embeddings, sim_matrix, embed_model, goal_text)
            labels = np.array([1.0 if nid in gold else 0.0 for nid in node_ids])
            X_train.append(feats[:, 1:])  # persistent only for gate
            y_train.append(labels)
        
        X_tr = np.vstack(X_train)
        y_tr = np.concatenate(y_train)
        
        if len(np.unique(y_tr)) >= 2:
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X_tr, y_tr)
            theta_gate = clf.coef_[0]
        else:
            theta_gate = np.ones(6) * 0.5
        
        # Linear theta (all 7 features)
        X_lin = np.vstack([compute_features(nodes, q["text"], query_time, embeddings, sim_matrix, embed_model, goal_text)[0] for q in train_qs])
        y_lin = np.concatenate([np.array([1.0 if nid in set(q["gold"]) else 0.0 for nid in node_ids]) for q in train_qs])
        if len(np.unique(y_lin)) >= 2:
            clf2 = LogisticRegression(max_iter=2000, C=1.0)
            clf2.fit(X_lin, y_lin)
            theta_linear = clf2.coef_[0]
        else:
            theta_linear = np.ones(7) * 0.5
        
        all_theta_gate.append(theta_gate)
        
        # Evaluate test queries
        for q in test_qs:
            gold = set(q["gold"])
            if not gold:
                continue
            feats, _ = compute_features(nodes, q["text"], query_time, embeddings, sim_matrix, embed_model, goal_text)
            
            for method in ["recency", "sim", "ga", "mga_gate", "mga_linear"]:
                if method == "mga_gate":
                    scores = retrieve(feats, method, theta_gate)
                elif method == "mga_linear":
                    scores = retrieve(feats, method, theta_linear)
                else:
                    scores = retrieve(feats, method)
                
                top_idx = np.argsort(scores)[::-1][:K]
                retrieved = [node_ids[i] for i in top_idx]
                
                recall = len(set(retrieved) & gold) / len(gold)
                precision = len(set(retrieved) & gold) / K
                stale_k = len(set(retrieved) & stale_set) / K
                noise_k = len(set(retrieved) & noise_set) / K
                
                # nDCG
                dcg = sum(1.0/np.log2(i+2) for i, nid in enumerate(retrieved) if nid in gold)
                idcg = sum(1.0/np.log2(i+2) for i in range(min(len(gold), K)))
                ndcg = dcg / max(idcg, 1e-8)
                
                all_results.append({
                    "seed": seed, "retriever": method, "query_id": q["id"],
                    "family": q["family"], "recall": recall, "precision": precision,
                    "ndcg": ndcg, "stale": stale_k, "noise": noise_k,
                })
    
    # ========== REPORT ==========
    df = pd.DataFrame(all_results)
    
    print(f"\n{'='*70}")
    print(f"  MGA BENCHMARK — FULL RESULTS (sentence-transformers, {N_WORLDS} worlds)")
    print(f"{'='*70}")
    print(f"  Total test queries: {len(df)}")
    
    print(f"\n  {'Retriever':<12} {'Recall@5':>10} {'nDCG@5':>10} {'Stale@5':>10} {'Noise@5':>10}")
    print(f"  {'-'*54}")
    for r in ["recency", "sim", "ga", "mga_gate", "mga_linear"]:
        rd = df[df["retriever"] == r]
        print(f"  {r:<12} {rd['recall'].mean():>10.4f} {rd['ndcg'].mean():>10.4f} {rd['stale'].mean():>10.4f} {rd['noise'].mean():>10.4f}")
    
    print(f"\n  PER-FAMILY BREAKDOWN (Recall@5):")
    print(f"  {'Family':<12}", end="")
    for r in ["recency", "sim", "ga", "mga_gate", "mga_linear"]:
        print(f" {r:>10}", end="")
    print()
    for fam in sorted(df["family"].unique()):
        fd = df[df["family"] == fam]
        print(f"  {fam:<12}", end="")
        for r in ["recency", "sim", "ga", "mga_gate", "mga_linear"]:
            rfd = fd[fd["retriever"] == r]
            print(f" {rfd['recall'].mean():>10.4f}" if len(rfd) > 0 else f" {'N/A':>10}", end="")
        print()
    
    print(f"\n  LEARNED THETA (mean across {N_WORLDS} worlds):")
    feat_names = ["recency", "frequency", "unresolved", "utility", "weight", "goal_rel"]
    mean_t = np.mean(all_theta_gate, axis=0)
    for name, val in zip(feat_names, mean_t):
        print(f"    {name:<15}: {val:>8.4f}")
    
    return df.to_dict()


@app.local_entrypoint()
def main():
    result = run_mga_benchmark.remote()
    print("\nBenchmark complete. Results returned.")
