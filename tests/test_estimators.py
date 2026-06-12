"""
Tests for signal estimators.
Validates recency decay, frequency counting, unresolved detection, etc.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from datetime import datetime, timedelta

from estimators import (
    estimate_recency, estimate_frequency, estimate_unresolved,
)


@pytest.fixture
def sample_nodes():
    """Create sample nodes spanning 10 days."""
    base = datetime(2024, 1, 1, 9, 0, 0)
    nodes = [
        {"id": "n0", "content": "We need to fix the bug in section 4.", "timestamp": (base + timedelta(days=0)).isoformat()},
        {"id": "n1", "content": "The citation style is APA 7.", "timestamp": (base + timedelta(days=2)).isoformat()},
        {"id": "n2", "content": "Had great coffee today.", "timestamp": (base + timedelta(days=4)).isoformat()},
        {"id": "n3", "content": "Still need to fix the bug in section 4.", "timestamp": (base + timedelta(days=6)).isoformat()},
        {"id": "n4", "content": "The bug in section 4 is fixed now.", "timestamp": (base + timedelta(days=8)).isoformat()},
        {"id": "n5", "content": "We still need to add references.", "timestamp": (base + timedelta(days=9)).isoformat()},
    ]
    return nodes


@pytest.fixture
def query_time():
    return (datetime(2024, 1, 1, 9, 0, 0) + timedelta(days=10)).isoformat()


class TestRecency:
    def test_recent_nodes_higher(self, sample_nodes, query_time):
        recency = estimate_recency(sample_nodes, query_time)
        # Node 5 (day 9) should be more recent than node 0 (day 0)
        assert recency[5] > recency[0]

    def test_exponential_decay(self, sample_nodes, query_time):
        recency = estimate_recency(sample_nodes, query_time, tau_r=5.0)
        # All values should be in (0, 1]
        assert np.all(recency > 0)
        assert np.all(recency <= 1.0)

    def test_monotonically_decreasing_with_age(self, query_time):
        """Older nodes should have lower recency."""
        base = datetime(2024, 1, 1, 9, 0, 0)
        nodes = [
            {"id": f"n{i}", "content": f"msg {i}", "timestamp": (base + timedelta(days=i)).isoformat()}
            for i in range(10)
        ]
        recency = estimate_recency(nodes, query_time)
        # Each node should have higher recency than the one before
        for i in range(1, len(recency)):
            assert recency[i] > recency[i - 1]

    def test_tau_r_controls_decay_rate(self, sample_nodes, query_time):
        fast_decay = estimate_recency(sample_nodes, query_time, tau_r=1.0)
        slow_decay = estimate_recency(sample_nodes, query_time, tau_r=20.0)
        # With fast decay, old nodes (node 0) are much more suppressed
        assert fast_decay[0] < slow_decay[0]
        # Recent nodes less affected
        assert abs(fast_decay[5] - slow_decay[5]) < abs(fast_decay[0] - slow_decay[0])


class TestFrequency:
    def test_repeated_content_higher_frequency(self):
        """Nodes with near-duplicates should have higher frequency."""
        # Create a similarity matrix where nodes 0,3 are similar
        n = 6
        sim_matrix = np.eye(n) * 1.0  # self-similarity = 1
        sim_matrix[0, 3] = 0.85  # near-duplicate
        sim_matrix[3, 0] = 0.85
        
        nodes = [{"id": f"n{i}", "content": f"msg {i}"} for i in range(n)]
        freq = estimate_frequency(nodes, sim_matrix, threshold=0.7)
        
        # Nodes 0 and 3 should have higher frequency than others
        assert freq[0] > freq[2]
        assert freq[3] > freq[2]

    def test_unique_content_zero_frequency(self):
        """Node with no near-duplicates should have 0 frequency."""
        n = 5
        sim_matrix = np.eye(n)  # no similarity between any pair
        nodes = [{"id": f"n{i}", "content": f"msg {i}"} for i in range(n)]
        freq = estimate_frequency(nodes, sim_matrix, threshold=0.7)
        
        np.testing.assert_array_equal(freq, np.zeros(n))

    def test_normalized_to_01(self):
        """Frequency should be in [0, 1]."""
        n = 10
        sim_matrix = np.random.rand(n, n)
        sim_matrix = (sim_matrix + sim_matrix.T) / 2
        np.fill_diagonal(sim_matrix, 1.0)
        nodes = [{"id": f"n{i}", "content": f"msg {i}"} for i in range(n)]
        freq = estimate_frequency(nodes, sim_matrix, threshold=0.7)
        
        assert np.all(freq >= 0)
        assert np.all(freq <= 1.0)


class TestUnresolved:
    def test_open_task_without_closure(self):
        """Task node with no closure should be unresolved=1."""
        nodes = [
            {"id": "n0", "content": "We still need to fix the bug."},
            {"id": "n1", "content": "Had coffee today."},
            {"id": "n2", "content": "The weather is nice."},
        ]
        # Low similarity matrix (no closure match)
        sim_matrix = np.eye(3) * 0.1
        np.fill_diagonal(sim_matrix, 1.0)
        
        unresolved = estimate_unresolved(nodes, sim_matrix)
        assert unresolved[0] == 1.0  # open task, no closure
        assert unresolved[1] == 0.0  # not a task
        assert unresolved[2] == 0.0  # not a task

    def test_closed_task_resolved(self):
        """Task followed by closure should be unresolved=0."""
        nodes = [
            {"id": "n0", "content": "We still need to fix the bug."},
            {"id": "n1", "content": "The bug is fixed now."},
        ]
        # High similarity between task and closure
        sim_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
        
        unresolved = estimate_unresolved(nodes, sim_matrix)
        assert unresolved[0] == 0.0  # closed!

    def test_non_task_always_zero(self):
        """Non-task nodes should always be unresolved=0."""
        nodes = [
            {"id": "n0", "content": "The weather is great today."},
            {"id": "n1", "content": "I had pizza for lunch."},
        ]
        sim_matrix = np.eye(2)
        unresolved = estimate_unresolved(nodes, sim_matrix)
        np.testing.assert_array_equal(unresolved, [0.0, 0.0])
