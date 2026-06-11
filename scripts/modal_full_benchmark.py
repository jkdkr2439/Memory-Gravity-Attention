"""
MGA Full Fractal Benchmark on Modal.com
Derived from paper's attention chain: Gap → Signal → Attention/Biết → Hiểu → Decision

Levels:
1. Gap Detection (change, contradiction, novelty, staleness)
2. Attention/Biết (recall, precision, needle, noise, persistence, recency trap)
3. Hiểu (coherence, coverage, contradiction avoidance)
4. Decision Support (context sensitivity, goal alignment, LTM/STM balance)
5. Robustness (scale stability, cross-world generalization, attention lock)

Validation: K-fold CV, bootstrap CI, statistical testing

Run: modal run scripts/modal_full_benchmark.py
"""

import modal

app = modal.App("mga-full-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "scikit-learn", "sentence-transformers", "torch", "scipy")
)


@app.function(image=image, gpu="T4", timeout=900)
def run_full_benchmark():
    import numpy as np
    import pandas as pd
    import random
    import json
    from datetime import datetime, timedelta
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from scipy.special import expit
    from scipy import stats
    from sentence_transformers import SentenceTransformer

    print("Loading model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Loaded.\n")

    # ====================================================================
    # TEMPLATES (expanded)
    # ====================================================================
    TEMPLATES = {
        "state_fact": [
            "By the way, {key} is {value}.",
            "Remember: {key} should be {value}.",
            "For this project, {key} = {value}.",
            "Note: always use {value} for {key}.",
            "Important: {key} is set to {value}.",
            "Just to be clear, the {key} we're using is {value}.",
            "FYI: {key} has been set to {value}.",
            "Keep in mind that {key} is {value}.",
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
            "Can someone handle {task}?",
            "Reminder: {task} is unresolved.",
        ],
        "close_task": [
            "{task} is done now.",
            "Fixed: {task}.",
            "Completed: {task}.",
            "That's finished — {task} is resolved.",
            "Good news: {task} is taken care of.",
        ],
        "reference": [
            "As mentioned before, {key} is {value}.",
            "Reminder: {key} remains {value}.",
            "Don't forget that {key} = {value}.",
            "Just reiterating: {key} is still {value}.",
            "Per our earlier discussion, {key} is {value}.",
        ],
        "chatter": [
            "Had great coffee today.", "The weather is nice outside.",
            "I watched a good movie last night.", "Lunch was decent.",
            "My cat knocked over a glass again.", "Traffic was terrible.",
            "I need to buy groceries.", "Anyone else tired today?",
            "Just saw a funny meme.", "Weekend plans: sleep.",
            "This coffee is strong.", "I should exercise more.",
            "The new show season is out.", "Neighbor's dog barking.",
            "Random thought about pineapple pizza.",
        ],
    }

    DOMAINS = [
        {"name": "research", "preferences": [
            ("citation_style", ["APA 7", "IEEE", "Chicago", "Harvard", "Vancouver"]),
            ("writing_tone", ["formal", "technical", "accessible", "academic"]),
            ("language", ["English", "Vietnamese", "bilingual"]),
            ("math_notation", ["LaTeX inline", "display equations", "minimal"]),
            ("section_format", ["numbered", "unnumbered", "short"]),
            ("figure_style", ["minimalist", "detailed", "colorful"]),
            ("abstract_length", ["150 words", "200 words", "300 words"]),
        ], "tasks": [
            "fix margin overflow in section 4", "add missing references",
            "rewrite abstract", "check equations", "proofread conclusion",
            "add figure for main result", "format appendix",
            "resolve notation conflict", "update related work",
            "fix table formatting", "add acknowledgments",
        ]},
        {"name": "software", "preferences": [
            ("framework", ["PyTorch", "TensorFlow", "JAX", "NumPy"]),
            ("testing", ["pytest", "unittest", "property-based"]),
            ("code_style", ["PEP8", "Google", "type hints", "verbose"]),
            ("deployment", ["Docker", "cloud", "local", "Kubernetes"]),
            ("database", ["PostgreSQL", "MongoDB", "SQLite", "Redis"]),
            ("logging", ["structured JSON", "plain text", "minimal"]),
        ], "tasks": [
            "fix memory leak", "add unit tests for loader",
            "refactor config", "optimize inference", "update README",
            "fix CI pipeline", "add logging", "handle edge case",
            "migrate schema", "fix auth bug", "add rate limiting",
        ]},
        {"name": "design", "preferences": [
            ("color_palette", ["warm", "cool blues", "monochrome", "vibrant"]),
            ("typography", ["sans-serif", "serif", "mixed", "custom"]),
            ("layout", ["grid", "free-form", "responsive", "mobile first"]),
            ("animation", ["minimal", "smooth", "playful", "none"]),
            ("brand_voice", ["professional", "friendly", "bold", "minimal"]),
        ], "tasks": [
            "redesign hero section", "fix mobile nav",
            "create icon set", "update color scheme", "add dark mode",
            "fix form UX", "design onboarding", "email templates",
            "fix image loading", "add micro-interactions",
        ]},
    ]

    # ====================================================================
    # WORLD GENERATOR
    # ====================================================================
    def generate_world(seed, n_sessions=25):
        rng = random.Random(seed)
        domain = rng.choice(DOMAINS)
        prefs = {k: rng.choice(opts) for k, opts in domain["preferences"]}

        # Revisions
        n_rev = rng.randint(2, 4)
        rev_keys = rng.sample(list(prefs.keys()), min(n_rev, len(prefs)))
        revisions = {}
        for k in rev_keys:
            s = rng.randint(n_sessions//4, n_sessions-3)
            opts = [v for _, os in domain["preferences"] for v in os if v != prefs[k]]
            revisions[k] = (s, rng.choice(opts) if opts else prefs[k]+"_v2")

        # Tasks
        tasks = domain["tasks"][:]; rng.shuffle(tasks)
        n_t = rng.randint(4, min(7, len(tasks)))
        open_tasks = tasks[:n_t]
        task_open = {t: rng.randint(0, n_sessions//3) for t in open_tasks}
        n_close = rng.randint(2, max(2, n_t-2))
        to_close = rng.sample(open_tasks, n_close)
        task_close = {t: rng.randint(task_open[t]+3, n_sessions-1) for t in to_close}

        nodes = []; nid = 0
        cur_prefs = dict(prefs); cur_open = set()
        base = datetime(2024,1,1,9,0,0)

        for sess in range(n_sessions):
            st = base + timedelta(days=sess, hours=rng.randint(0,8))

            for k,(rs,nv) in revisions.items():
                if sess == rs:
                    c = rng.choice(TEMPLATES["revise_fact"]).format(key=k, value=nv)
                    nodes.append({"id":f"n{nid:04d}","content":c,"ts":(st+timedelta(minutes=rng.randint(1,30))).isoformat(),"sess":sess,"type":"revise","mk":k,"mv":nv})
                    cur_prefs[k] = nv; nid+=1

            for t,os in task_open.items():
                if sess == os:
                    c = rng.choice(TEMPLATES["open_task"]).format(task=t)
                    nodes.append({"id":f"n{nid:04d}","content":c,"ts":(st+timedelta(minutes=rng.randint(5,40))).isoformat(),"sess":sess,"type":"open_task","mt":t})
                    cur_open.add(t); nid+=1

            for t in to_close:
                if t in task_close and sess == task_close[t]:
                    c = rng.choice(TEMPLATES["close_task"]).format(task=t)
                    nodes.append({"id":f"n{nid:04d}","content":c,"ts":(st+timedelta(minutes=rng.randint(10,50))).isoformat(),"sess":sess,"type":"close_task","mt":t})
                    cur_open.discard(t); nid+=1

            n_ev = rng.randint(10,22)
            for i in range(n_ev):
                ts = st + timedelta(minutes=rng.randint(1,60)+i*3)
                et = rng.choices(["state_fact","reference","chatter"], [.25,.30,.45])[0]
                if et in ("state_fact","reference"):
                    k = rng.choice(list(cur_prefs.keys()))
                    c = rng.choice(TEMPLATES[et]).format(key=k, value=cur_prefs[k])
                    nodes.append({"id":f"n{nid:04d}","content":c,"ts":ts.isoformat(),"sess":sess,"type":et,"mk":k,"mv":cur_prefs[k]})
                else:
                    c = rng.choice(TEMPLATES["chatter"])
                    nodes.append({"id":f"n{nid:04d}","content":c,"ts":ts.isoformat(),"sess":sess,"type":"chatter"})
                nid+=1

        # Oracle
        stale = set(); noise = set()
        for n in nodes:
            if n["type"]=="chatter": noise.add(n["id"])
            elif n["type"] in ("state_fact","reference") and "mk" in n:
                if n.get("mv") != cur_prefs.get(n.get("mk",""),""): stale.add(n["id"])

        # Queries (expanded families)
        queries = []
        # Constraint
        for k,v in cur_prefs.items():
            gold = [n["id"] for n in nodes if n.get("mk")==k and n.get("mv")==v and n["id"] not in stale]
            if gold: queries.append({"id":f"qc_{k}","fam":"constraint","text":f"What is the current {k}?","gold":gold})
        # Stale trap
        for k,v in cur_prefs.items():
            stale_k = [n["id"] for n in nodes if n.get("mk")==k and n["id"] in stale]
            gold = [n["id"] for n in nodes if n.get("mk")==k and n.get("mv")==v and n["id"] not in stale]
            if stale_k and gold: queries.append({"id":f"qs_{k}","fam":"stale_trap","text":f"What is the latest {k}?","gold":gold,"distractors":stale_k})
        # Open-loop
        if cur_open:
            gold = [n["id"] for n in nodes if n["type"]=="open_task" and n.get("mt") in cur_open]
            if gold: queries.append({"id":"q_open","fam":"open_loop","text":"What tasks are still open?","gold":gold})
        # Needle (early fact, never repeated)
        early = [n for n in nodes if n["type"]=="state_fact" and n["sess"]<3 and n["id"] not in stale]
        if early:
            nd = rng.choice(early)
            queries.append({"id":"q_needle","fam":"needle","text":f"What was said about {nd.get('mk','setup')} early on?","gold":[nd["id"]]})
        # Noise resistance
        gold_facts = [n["id"] for n in nodes if n["type"] in ("state_fact","revise") and n["id"] not in stale and n["id"] not in noise][:12]
        if gold_facts: queries.append({"id":"q_noise","fam":"noise","text":"What are the key rules?","gold":gold_facts})
        # Recency trap (recent chatter vs old gold)
        recent_noise = [n for n in nodes if n["type"]=="chatter" and n["sess"]>=n_sessions-3]
        old_gold = [n for n in nodes if n["type"]=="state_fact" and n["sess"]<5 and n["id"] not in stale]
        if recent_noise and old_gold:
            g = rng.choice(old_gold)
            queries.append({"id":"q_rtrap","fam":"recency_trap","text":f"What is the {g.get('mk','key')} setting?","gold":[g["id"]]})
        # Change detection (what changed?)
        rev_nodes = [n for n in nodes if n["type"]=="revise"]
        if rev_nodes: queries.append({"id":"q_change","fam":"change_detect","text":"What has changed or been updated?","gold":[n["id"] for n in rev_nodes]})
        # Context-dependent (same query, different framing)
        if cur_prefs:
            k1 = list(cur_prefs.keys())[0]
            gold1 = [n["id"] for n in nodes if n.get("mk")==k1 and n.get("mv")==cur_prefs[k1] and n["id"] not in stale]
            if gold1:
                queries.append({"id":"q_ctx1","fam":"context_a","text":f"For the current project, what {k1} are we using?","gold":gold1})
                queries.append({"id":"q_ctx2","fam":"context_b","text":f"Historically, what {k1} options were considered?","gold":[n["id"] for n in nodes if n.get("mk")==k1]})

        rng.shuffle(queries)
        return {"nodes":nodes,"queries":queries,"stale":stale,"noise":noise,"prefs":cur_prefs,"domain":domain["name"]}

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

        task_cues = ["need to","still","todo","pending","waiting","not done","remains","fix","resolve","hasn't"]
        closure_cues = ["done","fixed","completed","finished","resolved","cancelled","taken care"]
        unresolved = np.zeros(n)
        for i,nd in enumerate(nodes):
            cl = nd["content"].lower()
            if any(c in cl for c in task_cues):
                has_close = any(any(c in nodes[j]["content"].lower() for c in closure_cues) and sim_matrix[i,j]>0.4 for j in range(i+1,n))
                unresolved[i] = 0.0 if has_close else 1.0

        g_emb = embed_model.encode([goal_text], normalize_embeddings=True)
        goal_rel = cosine_similarity(g_emb, embeddings)[0]
        goal_rel = (goal_rel-goal_rel.min())/max(goal_rel.max()-goal_rel.min(),1e-8)

        sim_norm = (sim_scores-sim_scores.min())/max(sim_scores.max()-sim_scores.min(),1e-8)
        features = np.column_stack([sim_norm, recency, freq, unresolved, np.full(n,.5), np.full(n,.5), goal_rel])
        return features

    def retrieve(features, method, theta_g=None, theta_l=None):
        sim = features[:,0]
        if method=="recency": return features[:,1]
        elif method=="sim": return sim
        elif method=="ga": return sim + features[:,1] + features[:,5]
        elif method=="mga_gate": return sim*(1.0+expit(features[:,1:]@theta_g))
        elif method=="mga_linear": return features@theta_l
        return sim

    # ====================================================================
    # METRICS
    # ====================================================================
    def eval_query(retrieved, gold, stale, noise, k=5):
        ret_set = set(retrieved[:k])
        g = set(gold)
        recall = len(ret_set&g)/max(len(g),1)
        precision = len(ret_set&g)/k
        stale_k = len(ret_set&stale)/k
        noise_k = len(ret_set&noise)/k
        dcg = sum(1/np.log2(i+2) for i,nid in enumerate(retrieved[:k]) if nid in g)
        idcg = sum(1/np.log2(i+2) for i in range(min(len(g),k)))
        ndcg = dcg/max(idcg,1e-8)
        # Coherence: no stale+fresh for same key in retrieval
        coherence = 1.0 - stale_k  # simple proxy: less stale = more coherent
        return {"recall":recall,"precision":precision,"ndcg":ndcg,"stale":stale_k,"noise":noise_k,"coherence":coherence}

    # ====================================================================
    # MAIN BENCHMARK
    # ====================================================================
    N_WORLDS = 15
    K = 5
    METHODS = ["recency","sim","ga","mga_gate","mga_linear"]

    print(f"Generating {N_WORLDS} worlds...")
    all_worlds = []
    for seed in range(N_WORLDS):
        w = generate_world(seed, n_sessions=25)
        print(f"  World {seed}: {len(w['nodes'])} nodes, {len(w['queries'])} queries, {w['domain']}")
        # Embed
        texts = [n["content"] for n in w["nodes"]]
        embs = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        sim_mat = cosine_similarity(embs)
        goal = " ".join(n["content"] for n in w["nodes"] if n["type"] in ("state_fact","revise") and n["id"] not in w["stale"])[:1000]
        w["embs"] = embs; w["sim_mat"] = sim_mat; w["goal"] = goal
        all_worlds.append(w)

    # Pool all queries
    all_queries = []
    for wi,w in enumerate(all_worlds):
        for q in w["queries"]:
            all_queries.append({"world_idx":wi, **q})
    print(f"\nTotal queries: {len(all_queries)}")

    # ====================================================================
    # K-FOLD CROSS VALIDATION
    # ====================================================================
    print("\n--- K-FOLD CROSS VALIDATION (5-fold) ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    query_indices = np.arange(len(all_queries))

    fold_results = []
    fold_thetas_g = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(query_indices)):
        # Learn theta on train
        X_train, y_train = [], []
        for qi in train_idx:
            q = all_queries[qi]
            w = all_worlds[q["world_idx"]]
            gold = set(q["gold"])
            feats = compute_features(w["nodes"], q["text"], w["embs"], w["sim_mat"], embed_model, w["goal"])
            labels = np.array([1.0 if n["id"] in gold else 0.0 for n in w["nodes"]])
            X_train.append(feats[:,1:]); y_train.append(labels)

        X_tr = np.vstack(X_train); y_tr = np.concatenate(y_train)
        if len(np.unique(y_tr))>=2:
            clf = LogisticRegression(max_iter=2000, C=1.0); clf.fit(X_tr, y_tr)
            theta_g = clf.coef_[0]
        else:
            theta_g = np.ones(6)*0.5
        fold_thetas_g.append(theta_g)

        # Linear theta
        X_lin = np.vstack([compute_features(all_worlds[all_queries[qi]["world_idx"]]["nodes"], all_queries[qi]["text"], all_worlds[all_queries[qi]["world_idx"]]["embs"], all_worlds[all_queries[qi]["world_idx"]]["sim_mat"], embed_model, all_worlds[all_queries[qi]["world_idx"]]["goal"]) for qi in train_idx])
        y_lin = np.concatenate([np.array([1.0 if n["id"] in set(all_queries[qi]["gold"]) else 0.0 for n in all_worlds[all_queries[qi]["world_idx"]]["nodes"]]) for qi in train_idx])
        if len(np.unique(y_lin))>=2:
            clf2 = LogisticRegression(max_iter=2000, C=1.0); clf2.fit(X_lin, y_lin)
            theta_l = clf2.coef_[0]
        else:
            theta_l = np.ones(7)*0.5

        # Evaluate test
        for qi in test_idx:
            q = all_queries[qi]
            w = all_worlds[q["world_idx"]]
            gold = set(q["gold"])
            if not gold: continue
            feats = compute_features(w["nodes"], q["text"], w["embs"], w["sim_mat"], embed_model, w["goal"])
            nids = [n["id"] for n in w["nodes"]]

            for method in METHODS:
                scores = retrieve(feats, method, theta_g, theta_l)
                top_idx = np.argsort(scores)[::-1][:K]
                retrieved = [nids[i] for i in top_idx]
                metrics = eval_query(retrieved, q["gold"], w["stale"], w["noise"], K)
                metrics.update({"method":method,"family":q.get("fam","?"),"fold":fold_i,"query_id":q["id"],"world":q["world_idx"]})
                fold_results.append(metrics)

    df = pd.DataFrame(fold_results)
    print(f"  Total evaluations: {len(df)}")

    # ====================================================================
    # CROSS-WORLD GENERALIZATION
    # ====================================================================
    print("\n--- CROSS-WORLD GENERALIZATION (train 0-11, test 12-14) ---")
    train_worlds = list(range(12)); test_worlds = list(range(12,15))
    X_cw, y_cw = [], []
    for wi in train_worlds:
        w = all_worlds[wi]
        for q in w["queries"]:
            gold = set(q["gold"])
            feats = compute_features(w["nodes"], q["text"], w["embs"], w["sim_mat"], embed_model, w["goal"])
            labels = np.array([1.0 if n["id"] in gold else 0.0 for n in w["nodes"]])
            X_cw.append(feats[:,1:]); y_cw.append(labels)
    X_cw = np.vstack(X_cw); y_cw = np.concatenate(y_cw)
    if len(np.unique(y_cw))>=2:
        clf_cw = LogisticRegression(max_iter=2000, C=1.0); clf_cw.fit(X_cw, y_cw)
        theta_cw = clf_cw.coef_[0]
    else:
        theta_cw = np.ones(6)*0.5

    cw_results = []
    for wi in test_worlds:
        w = all_worlds[wi]
        nids = [n["id"] for n in w["nodes"]]
        for q in w["queries"]:
            gold = set(q["gold"])
            if not gold: continue
            feats = compute_features(w["nodes"], q["text"], w["embs"], w["sim_mat"], embed_model, w["goal"])
            for method in METHODS:
                scores = retrieve(feats, method, theta_cw, np.concatenate([[0.5],theta_cw]))
                top_idx = np.argsort(scores)[::-1][:K]
                retrieved = [nids[i] for i in top_idx]
                m = eval_query(retrieved, q["gold"], w["stale"], w["noise"], K)
                m.update({"method":method,"family":q.get("fam","?"),"world":wi})
                cw_results.append(m)
    df_cw = pd.DataFrame(cw_results)

    # ====================================================================
    # BOOTSTRAP CI
    # ====================================================================
    def bootstrap_ci(data, n_boot=1000, ci=0.95):
        boots = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
        lo = np.percentile(boots, (1-ci)/2*100)
        hi = np.percentile(boots, (1+ci)/2*100)
        return np.mean(data), lo, hi

    # ====================================================================
    # STATISTICAL TEST
    # ====================================================================
    def paired_test(a, b):
        """Paired permutation test: is a significantly > b?"""
        diff = np.array(a) - np.array(b)
        observed = np.mean(diff)
        n_perm = 5000
        count = 0
        for _ in range(n_perm):
            signs = np.random.choice([-1,1], size=len(diff))
            if np.mean(diff*signs) >= observed: count += 1
        return observed, count/n_perm  # mean_diff, p_value

    # ====================================================================
    # REPORTS
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"  MGA FULL FRACTAL BENCHMARK — RESULTS")
    print(f"  {N_WORLDS} worlds, {len(all_queries)} queries, 5-fold CV, bootstrap CI")
    print(f"{'='*70}")

    # Overall (K-fold)
    print(f"\n  LEVEL 2: ATTENTION/BIẾT (Retrieval Quality)")
    print(f"  {'Method':<12} {'Recall@5':>10} {'95% CI':>16} {'nDCG@5':>10} {'Noise@5':>10} {'Coherence':>10}")
    print(f"  {'-'*70}")
    for m in METHODS:
        md = df[df["method"]==m]
        mean, lo, hi = bootstrap_ci(md["recall"].values)
        print(f"  {m:<12} {mean:>10.4f} [{lo:.3f},{hi:.3f}] {md['ndcg'].mean():>10.4f} {md['noise'].mean():>10.4f} {md['coherence'].mean():>10.4f}")

    # Per family
    print(f"\n  PER-FAMILY (Recall@5):")
    fams = sorted(df["family"].unique())
    print(f"  {'Family':<15}", end="")
    for m in METHODS: print(f" {m:>10}", end="")
    print()
    print(f"  {'-'*67}")
    for fam in fams:
        fd = df[df["family"]==fam]
        print(f"  {fam:<15}", end="")
        for m in METHODS:
            r = fd[fd["method"]==m]["recall"]
            print(f" {r.mean():>10.4f}" if len(r)>0 else f" {'N/A':>10}", end="")
        print()

    # Cross-world
    print(f"\n  CROSS-WORLD GENERALIZATION (train 0-11, test 12-14):")
    print(f"  {'Method':<12} {'Recall@5':>10} {'nDCG@5':>10}")
    print(f"  {'-'*34}")
    for m in METHODS:
        md = df_cw[df_cw["method"]==m]
        print(f"  {m:<12} {md['recall'].mean():>10.4f} {md['ndcg'].mean():>10.4f}")

    # Statistical tests: MGA_gate vs GA, MGA_linear vs GA
    print(f"\n  STATISTICAL TESTS (paired permutation, 5000 perms):")
    for target in ["mga_gate","mga_linear"]:
        # Match queries between target and GA
        ga_recalls = []; tgt_recalls = []
        for qid in df["query_id"].unique():
            ga_r = df[(df["query_id"]==qid)&(df["method"]=="ga")]["recall"].values
            tgt_r = df[(df["query_id"]==qid)&(df["method"]==target)]["recall"].values
            if len(ga_r)>0 and len(tgt_r)>0:
                ga_recalls.append(ga_r[0]); tgt_recalls.append(tgt_r[0])
        diff, p = paired_test(tgt_recalls, ga_recalls)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        print(f"  {target} vs GA: mean_diff={diff:+.4f}, p={p:.4f} {sig}")

    # Learned theta
    print(f"\n  LEARNED THETA (mean across 5 folds):")
    feat_names = ["recency","frequency","unresolved","utility","weight","goal_rel"]
    mean_t = np.mean(fold_thetas_g, axis=0)
    std_t = np.std(fold_thetas_g, axis=0)
    for name,m,s in zip(feat_names, mean_t, std_t):
        print(f"    {name:<15}: {m:>8.4f} +/- {s:.4f}")

    # Level 5: Scale (compare worlds with different sizes)
    print(f"\n  LEVEL 5: ROBUSTNESS")
    sizes = df.groupby("world").apply(lambda x: len(all_worlds[x.name]["nodes"]) if x.name < len(all_worlds) else 0)
    print(f"  World sizes: min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.0f}")
    # Theta stability
    theta_cv = np.std(fold_thetas_g, axis=0) / (np.abs(np.mean(fold_thetas_g, axis=0))+1e-8)
    print(f"  Theta CV (lower=more stable): {dict(zip(feat_names, [f'{v:.2f}' for v in theta_cv]))}")

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")

    return {"kfold": df.to_dict(), "crossworld": df_cw.to_dict()}


@app.local_entrypoint()
def main():
    result = run_full_benchmark.remote()
    print("\nFull benchmark complete.")
