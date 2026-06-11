"""
Synthetic Interaction-Log Generator (§3 of spec)

Generates multi-session interaction logs with:
- Personas with preferences, projects, tasks
- Revision schedule (stale traps)
- Event types: state_fact, revise_fact, open_task, close_task, reference, chatter, use_episode
- Hidden world state (oracle) for evaluation only
"""

import json
import random
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple


TEMPLATES = {
    "state_fact": [
        "By the way, {key} is {value}.",
        "Remember: {key} should be {value}.",
        "For this project, {key} = {value}.",
        "Note: always use {value} for {key}.",
        "Important: {key} is set to {value}.",
    ],
    "revise_fact": [
        "Actually, change {key} to {value} now.",
        "Update: {key} is now {value} instead.",
        "Correction: switch {key} to {value}.",
        "New rule: {key} should be {value} going forward.",
    ],
    "open_task": [
        "We still need to {task}.",
        "TODO: {task}.",
        "Pending: {task} is not done yet.",
        "Don't forget — {task} remains open.",
        "Still waiting on: {task}.",
    ],
    "close_task": [
        "{task} is done now.",
        "Fixed: {task}.",
        "Completed: {task}.",
        "That's finished — {task} is resolved.",
    ],
    "reference": [
        "As mentioned before, {key} is {value}.",
        "Reminder: {key} remains {value}.",
        "Don't forget that {key} = {value}.",
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
    ],
}

# Sample knowledge domains for synthetic personas
DOMAINS = [
    {
        "name": "research_paper",
        "preferences": [
            ("citation_style", ["APA 7", "IEEE", "Chicago", "Harvard"]),
            ("writing_tone", ["formal", "technical", "accessible", "academic"]),
            ("language", ["English", "Vietnamese", "bilingual"]),
            ("math_notation", ["LaTeX inline", "display equations", "minimal math"]),
            ("section_format", ["numbered", "unnumbered", "short sections"]),
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
        ],
    },
    {
        "name": "software_project",
        "preferences": [
            ("framework", ["PyTorch", "TensorFlow", "JAX", "NumPy only"]),
            ("testing", ["pytest", "unittest", "no tests", "property-based"]),
            ("code_style", ["PEP8", "Google style", "minimal comments", "verbose docs"]),
            ("deployment", ["Docker", "bare metal", "cloud functions", "local only"]),
            ("version_control", ["git flow", "trunk-based", "feature branches"]),
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
        ],
    },
]


def generate_world(seed: int, n_sessions: int = 12, events_per_session: Tuple[int,int] = (8, 20)):
    """Generate a complete synthetic world with interaction logs and oracle."""
    rng = random.Random(seed)
    
    # Pick domain
    domain = rng.choice(DOMAINS)
    
    # Initialize persona
    preferences = {}
    for key, options in domain["preferences"]:
        preferences[key] = rng.choice(options)
    
    # Plan revisions (some facts change mid-log)
    n_revisions = rng.randint(1, 3)
    revision_keys = rng.sample(list(preferences.keys()), min(n_revisions, len(preferences)))
    revision_schedule = {}
    for key in revision_keys:
        revision_session = rng.randint(n_sessions // 3, n_sessions - 2)
        options = [v for _, opts in domain["preferences"] for v in opts if v != preferences[key]]
        new_value = rng.choice(options) if options else preferences[key] + "_v2"
        revision_schedule[key] = (revision_session, new_value)
    
    # Plan tasks
    all_tasks = domain["tasks"][:]
    rng.shuffle(all_tasks)
    n_tasks = rng.randint(3, min(6, len(all_tasks)))
    open_tasks = all_tasks[:n_tasks]
    task_open_session = {t: rng.randint(0, n_sessions // 2) for t in open_tasks}
    # Some tasks get closed
    n_close = rng.randint(1, max(1, n_tasks - 1))
    tasks_to_close = rng.sample(open_tasks, n_close)
    task_close_session = {t: rng.randint(task_open_session[t] + 2, n_sessions - 1) for t in tasks_to_close}
    
    # Generate event stream
    nodes = []
    events = []
    world_states = []  # oracle: state at each timestamp
    
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    node_id = 0
    current_prefs = dict(preferences)
    current_open_tasks = set()
    
    for session in range(n_sessions):
        session_time = base_time + timedelta(days=session, hours=rng.randint(0, 8))
        n_events = rng.randint(*events_per_session)
        
        # Check if any revision happens this session
        for key, (rev_session, new_val) in revision_schedule.items():
            if session == rev_session:
                # Emit revise event
                template = rng.choice(TEMPLATES["revise_fact"])
                content = template.format(key=key, value=new_val)
                ts = session_time + timedelta(minutes=rng.randint(1, 30))
                nodes.append({
                    "id": f"node_{node_id:04d}",
                    "content": content,
                    "timestamp": ts.isoformat(),
                    "session_id": session,
                    "event_type": "revise_fact",
                })
                events.append({"type": "revise_fact", "node_id": f"node_{node_id:04d}", "key": key, "old_value": current_prefs[key], "new_value": new_val})
                current_prefs[key] = new_val
                node_id += 1
        
        # Check task openings
        for task, open_s in task_open_session.items():
            if session == open_s:
                template = rng.choice(TEMPLATES["open_task"])
                content = template.format(task=task)
                ts = session_time + timedelta(minutes=rng.randint(5, 40))
                nodes.append({
                    "id": f"node_{node_id:04d}",
                    "content": content,
                    "timestamp": ts.isoformat(),
                    "session_id": session,
                    "event_type": "open_task",
                })
                events.append({"type": "open_task", "node_id": f"node_{node_id:04d}", "task": task})
                current_open_tasks.add(task)
                node_id += 1
        
        # Check task closings
        for task in tasks_to_close:
            if task in task_close_session and session == task_close_session[task]:
                template = rng.choice(TEMPLATES["close_task"])
                content = template.format(task=task)
                ts = session_time + timedelta(minutes=rng.randint(10, 50))
                nodes.append({
                    "id": f"node_{node_id:04d}",
                    "content": content,
                    "timestamp": ts.isoformat(),
                    "session_id": session,
                    "event_type": "close_task",
                })
                events.append({"type": "close_task", "node_id": f"node_{node_id:04d}", "task": task})
                current_open_tasks.discard(task)
                node_id += 1
        
        # Fill remaining events
        for i in range(n_events):
            ts = session_time + timedelta(minutes=rng.randint(1, 60) + i * 3)
            event_type = rng.choices(
                ["state_fact", "reference", "chatter"],
                weights=[0.3, 0.3, 0.4],
                k=1
            )[0]
            
            if event_type == "state_fact":
                key = rng.choice(list(current_prefs.keys()))
                template = rng.choice(TEMPLATES["state_fact"])
                content = template.format(key=key, value=current_prefs[key])
            elif event_type == "reference":
                key = rng.choice(list(current_prefs.keys()))
                template = rng.choice(TEMPLATES["reference"])
                content = template.format(key=key, value=current_prefs[key])
            else:
                content = rng.choice(TEMPLATES["chatter"])
            
            nodes.append({
                "id": f"node_{node_id:04d}",
                "content": content,
                "timestamp": ts.isoformat(),
                "session_id": session,
                "event_type": event_type,
            })
            node_id += 1
        
        # Record world state at end of session
        world_states.append({
            "session": session,
            "current_preferences": dict(current_prefs),
            "open_tasks": list(current_open_tasks),
            "revision_history": {k: (s, v) for k, (s, v) in revision_schedule.items() if s <= session},
        })
    
    # Generate oracle: which nodes are stale, noise, etc.
    oracle = {
        "final_preferences": current_prefs,
        "open_tasks_at_end": list(current_open_tasks),
        "stale_nodes": [],
        "noise_nodes": [],
        "task_nodes": [],
    }
    
    for node in nodes:
        if node["event_type"] == "chatter":
            oracle["noise_nodes"].append(node["id"])
        elif node["event_type"] == "open_task":
            # Check if task was later closed
            task_text = node["content"]
            oracle["task_nodes"].append(node["id"])
        elif node["event_type"] in ("state_fact", "reference"):
            # Check if this fact was later revised
            for key, (rev_s, new_val) in revision_schedule.items():
                if key in node["content"] and node["session_id"] < rev_s:
                    if current_prefs[key] != new_val:
                        continue
                    # This node states old value → stale
                    old_vals = [v for _, opts in domain["preferences"] for v in opts]
                    for ov in old_vals:
                        if ov in node["content"] and ov != current_prefs.get(key, ""):
                            oracle["stale_nodes"].append(node["id"])
                            break
    
    # Generate queries
    queries = generate_queries(current_prefs, current_open_tasks, nodes, oracle, rng)
    
    return {
        "nodes": nodes,
        "events": events,
        "oracle": oracle,
        "world_states": world_states,
        "queries": queries,
        "domain": domain["name"],
        "seed": seed,
    }


def generate_queries(prefs, open_tasks, nodes, oracle, rng) -> List[Dict]:
    """Generate test queries with gold labels."""
    queries = []
    
    # Constraint recall queries
    for key, value in prefs.items():
        queries.append({
            "id": f"q_constraint_{key}",
            "family": "constraint",
            "text": f"What is the current {key}?",
            "gold_nodes": [n["id"] for n in nodes 
                          if key in n["content"] and value in n["content"]
                          and n["id"] not in oracle["stale_nodes"]],
        })
    
    # Open task queries
    if open_tasks:
        queries.append({
            "id": "q_open_tasks",
            "family": "open_loop",
            "text": "What tasks are still open and unresolved?",
            "gold_nodes": [n["id"] for n in nodes
                          if n["event_type"] == "open_task"
                          and any(t in n["content"] for t in open_tasks)],
        })
    
    # Stale trap queries (asks for current value, stale node is distractor)
    for key, value in prefs.items():
        stale_for_key = [n["id"] for n in nodes 
                        if n["id"] in oracle["stale_nodes"] and key in n["content"]]
        if stale_for_key:
            queries.append({
                "id": f"q_stale_{key}",
                "family": "stale_trap",
                "text": f"What is the latest {key} we should use?",
                "gold_nodes": [n["id"] for n in nodes
                              if key in n["content"] and value in n["content"]
                              and n["id"] not in oracle["stale_nodes"]],
                "distractor_nodes": stale_for_key,
            })
    
    # Noise resistance query
    queries.append({
        "id": "q_noise_resist",
        "family": "noise",
        "text": "What are the key project constraints and rules?",
        "gold_nodes": [n["id"] for n in nodes
                      if n["event_type"] in ("state_fact", "revise_fact")
                      and n["id"] not in oracle["stale_nodes"]
                      and n["id"] not in oracle["noise_nodes"]],
    })
    
    rng.shuffle(queries)
    return queries


def save_world(world: Dict, output_dir: str):
    """Save world to disk with proper separation."""
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "oracle"), exist_ok=True)
    
    # Log plane (visible)
    with open(os.path.join(output_dir, "logs", "nodes.json"), "w", encoding="utf-8") as f:
        json.dump(world["nodes"], f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, "logs", "events.json"), "w", encoding="utf-8") as f:
        json.dump(world["events"], f, indent=2, ensure_ascii=False)
    
    # Oracle plane (evaluator only)
    with open(os.path.join(output_dir, "oracle", "world_state.json"), "w", encoding="utf-8") as f:
        json.dump(world["oracle"], f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, "oracle", "queries.json"), "w", encoding="utf-8") as f:
        json.dump(world["queries"], f, indent=2, ensure_ascii=False)
    
    # Metadata
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"domain": world["domain"], "seed": world["seed"], 
                   "n_nodes": len(world["nodes"]), "n_queries": len(world["queries"])}, f, indent=2)


if __name__ == "__main__":
    print("Generating synthetic worlds...")
    for seed in range(5):
        world = generate_world(seed=seed)
        output_dir = f"d:/Existence/MGA/mga_project/data/world_{seed}"
        save_world(world, output_dir)
        print(f"  World {seed}: {len(world['nodes'])} nodes, {len(world['queries'])} queries, domain={world['domain']}")
    print("Done.")
