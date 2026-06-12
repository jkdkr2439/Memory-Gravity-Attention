# Memory-Gravity Attention (MGA)

**A model-agnostic retrieval layer that adds persistent-importance signals to similarity-based memory retrieval.**

Author: Kevin T.N

---

## What is MGA?

Standard LLM memory retrieval uses similarity (what looks like the query?) and recency (what happened recently?). This works for most cases but **fails on tasks requiring long-horizon persistent memory**:

- Finding a fact stated once, long ago (needle)
- Tracking tasks that remain open/unresolved (open-loop)
- Filtering noise from signal in dense logs
- Avoiding stale/superseded information

MGA adds a **persistent-importance gate** to similarity:

```
score = similarity × (1 + σ(θᵀ · [recency, frequency, unresolved, utility, weight, goal_relevance]))
```

The gate amplifies similarity for nodes that have persistent importance, while ensuring that low-similarity nodes can't be dragged in by importance alone.

---

## Key Results

Benchmark: 10 synthetic worlds, 415 test queries, sentence-transformers embeddings, GPU (Modal T4).

| Retriever | Recall@5 | nDCG@5 | Stale@5 | Noise@5 |
|-----------|----------|--------|---------|---------|
| recency | 0.029 | 0.085 | 0.000 | 0.364 |
| similarity | 0.138 | 0.496 | 0.340 | 0.046 |
| GA (Generative Agents) | 0.209 | 0.697 | 0.000 | 0.070 |
| **MGA gate** | 0.174 | 0.562 | 0.265 | **0.029** |
| **MGA linear** | 0.195 | 0.614 | 0.205 | **0.017** |

### Where MGA wins (tasks that require persistent memory):

| Task Family | GA | MGA gate | MGA linear |
|-------------|-----|----------|------------|
| Needle (old fact, stated once) | 0.000 | **0.143** | 0.000 |
| Open-loop (unresolved tasks) | 0.000 | 0.111 | **0.222** |
| Noise resistance | 0.000 | **0.075** | 0.025 |

**GA scores 0.000 on needle, open-loop, and noise tasks.** These are precisely the tasks where persistent memory matters and recency/similarity alone fail.

---

## Learned Feature Importance

From logistic regression on training queries (mean across 10 worlds):

| Feature | Coefficient |
|---------|-------------|
| goal_relevance | **2.92** |
| recency | 0.67 |
| frequency | 0.40 |
| unresolved | 0.21 |
| utility | -0.01 |
| weight | -0.01 |

**Goal relevance is the strongest persistent signal** — knowing what the system is currently working on dramatically improves retrieval.

---

## Architecture

```
[Interaction Log] → [Signal Estimators] → [Retrievers] → [Evaluator]
     (visible)         (heuristic)         (5 methods)    (oracle only)
```

Three data planes (strict separation):
- **Log plane**: raw interaction events (visible to everything)
- **Estimated plane**: computed signals from logs (visible to retrievers)
- **Oracle plane**: gold labels, stale flags (visible ONLY to evaluator)

No retriever ever sees gold labels. Leakage-tested.

---

## Theoretical Connection

MGA is the engineering implementation of **Existential Attention** — attention based on gap × relevance rather than similarity alone:

- Standard attention: "What in context is SIMILAR to my query?"
- Existential attention: "What is DIFFERENT from my current state AND relevant to my survival?"

In memory retrieval terms:
- Similarity finds what LOOKS like the query
- MGA finds what MATTERS for the ongoing task (persistent importance)

See: [Formal Equation of Existence](https://github.com/jkdkr2439/What-If-AI-Wants-to-Destroy-Humanity)

---

## Project Structure

```
mga_project/
├── README.md
├── requirements.txt
├── src/
│   ├── generator.py           # Synthetic multi-session log generator
│   ├── estimators.py          # Signal estimation (recency, frequency, unresolved, goal_rel)
│   ├── similarity.py          # TF-IDF (toy) and sentence-transformers (real)
│   ├── retrievers.py          # 5 retrieval methods (recency, sim, GA, MGA gate, MGA linear)
│   ├── learn.py               # Logistic regression for theta
│   ├── metrics.py             # Recall@k, nDCG@k, Stale@k, Noise@k
│   └── entropy_diagnostic.py  # Attention-entropy lock detection
├── scripts/
│   ├── run_benchmark.py           # Local benchmark (CPU, TF-IDF)
│   ├── run_entropy_diagnostic.py  # Run lock detection analysis
│   ├── modal_benchmark.py         # Cloud benchmark (GPU, sentence-transformers)
│   ├── modal_full_benchmark.py    # Full fractal benchmark (15 worlds, K-fold, CI)
│   ├── modal_locomo_benchmark.py  # LoCoMo-MC10 real dataset benchmark
│   └── modal_realworld_benchmark.py # ShareGPT/OASST2 real data benchmark
├── tests/
│   ├── test_retrievers.py         # Retriever logic tests
│   ├── test_estimators.py         # Signal estimator tests
│   ├── test_metrics.py            # Metric calculation tests
│   ├── test_entropy_diagnostic.py # Lock detection tests
│   └── test_generator.py          # World generator tests
├── paper/
│   └── mga_paper.tex              # Full paper (LaTeX)
└── outputs/
    └── results.csv                # Raw results
```

---

## Quick Start

### Local (CPU, toy similarity):
```bash
pip install numpy pandas scikit-learn
python scripts/run_benchmark.py
```

### Entropy diagnostic (lock detection):
```bash
python scripts/run_entropy_diagnostic.py
```

### Cloud (GPU, real embeddings):
```bash
pip install modal
modal run scripts/modal_benchmark.py
```

### Real-world datasets:
```bash
modal run scripts/modal_locomo_benchmark.py      # LoCoMo-MC10
modal run scripts/modal_realworld_benchmark.py   # OASST2/ShareGPT
```

### Run tests:
```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Baselines

| Method | Source | Description |
|--------|--------|-------------|
| recency | standard | Rank by time only |
| similarity | standard | Rank by embedding cosine similarity only |
| GA | Park et al. 2023 | recency + importance + similarity (Generative Agents) |
| MGA gate | **this work** | similarity × (1 + σ(θᵀ · persistent_features)) |
| MGA linear | **this work** (ablation) | θᵀ · all_features |

---

## Honest Assessment

- **GA wins overall recall/nDCG** because recency naturally filters stale nodes and the benchmark has many recency-friendly queries.
- **MGA wins on the hard tasks** where persistent memory matters: needle recall, open-loop tracking, noise filtering.
- **Contribution**: MGA doesn't replace GA. It fills the gap where GA (and similarity-only) fail — long-horizon, persistent-importance retrieval.

---

## Attention-Entropy Diagnostic (Lock Detection)

Detects when a retriever is "locked" — attending to a narrow, repetitive set of nodes regardless of query content.

| Retriever | Mean Entropy | Std | Top-5 Concentration |
|-----------|:---:|:---:|:---:|
| recency | 0.992 | 0.001 | 0.050 |
| similarity | 0.992 | 0.005 | 0.060 |
| GA | 0.983 | 0.005 | 0.078 |
| **MGA gate** | 0.976 | 0.014 | 0.093 |
| **MGA linear** | 0.650 | 0.131 | 0.467 |

**Findings**: No retriever shows pathological lock. MGA linear concentrates attention most aggressively (top-5 concentration 0.467), explaining its stronger targeted recall. MGA gate maintains high entropy while still achieving selective boosting — the multiplicative structure amplifies without collapsing.

---

## Citation

```
@misc{mga2026,
  author = {Kevin T.N},
  title = {Memory-Gravity Attention: Persistent-Importance Retrieval for Long-Horizon Memory},
  year = {2026},
  url = {https://github.com/jkdkr2439/MGA}
}
```

---

## License

MIT

