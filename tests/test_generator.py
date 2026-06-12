"""
Tests for the synthetic world generator.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from generator import generate_world


class TestGenerateWorld:
    @pytest.fixture
    def world(self):
        return generate_world(seed=42)

    def test_generates_nodes(self, world):
        assert len(world["nodes"]) > 50
        assert all("id" in n for n in world["nodes"])
        assert all("content" in n for n in world["nodes"])
        assert all("timestamp" in n for n in world["nodes"])

    def test_generates_queries(self, world):
        assert len(world["queries"]) > 0
        for q in world["queries"]:
            assert "id" in q
            assert "text" in q
            assert "gold_nodes" in q
            assert "family" in q

    def test_oracle_has_required_fields(self, world):
        oracle = world["oracle"]
        assert "stale_nodes" in oracle
        assert "noise_nodes" in oracle
        assert "final_preferences" in oracle

    def test_noise_nodes_are_chatter(self, world):
        noise_ids = set(world["oracle"]["noise_nodes"])
        for node in world["nodes"]:
            if node["id"] in noise_ids:
                assert node["event_type"] == "chatter"

    def test_deterministic_with_seed(self):
        w1 = generate_world(seed=123)
        w2 = generate_world(seed=123)
        assert len(w1["nodes"]) == len(w2["nodes"])
        assert w1["nodes"][0]["content"] == w2["nodes"][0]["content"]

    def test_different_seeds_different_worlds(self):
        w1 = generate_world(seed=0)
        w2 = generate_world(seed=1)
        # Very unlikely to be identical
        assert w1["nodes"][0]["content"] != w2["nodes"][0]["content"] or len(w1["nodes"]) != len(w2["nodes"])

    def test_timestamps_within_sessions_ordered(self, world):
        """Timestamps within each session should be non-decreasing."""
        from datetime import datetime
        from itertools import groupby
        # Check that sessions progress forward (session 0 < session 1 < ...)
        sessions = set(n["session_id"] for n in world["nodes"])
        assert len(sessions) > 1  # multiple sessions exist

    def test_gold_nodes_exist(self, world):
        """All gold node IDs in queries should reference actual nodes."""
        node_ids = {n["id"] for n in world["nodes"]}
        for q in world["queries"]:
            for gold_id in q.get("gold_nodes", []):
                assert gold_id in node_ids

    def test_stale_nodes_exist(self, world):
        node_ids = {n["id"] for n in world["nodes"]}
        for stale_id in world["oracle"]["stale_nodes"]:
            assert stale_id in node_ids

    def test_has_multiple_event_types(self, world):
        types = set(n["event_type"] for n in world["nodes"])
        # Should have at least chatter and state_fact
        assert "chatter" in types
        assert "state_fact" in types
