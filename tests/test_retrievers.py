"""
Tests for MGA retrievers.
Validates core retrieval logic, score ordering, and gate behavior.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from scipy.special import expit

from retrievers import (
    retrieve_recency, retrieve_similarity, retrieve_ga,
    retrieve_mga_gate, retrieve_mga_linear, top_k, RETRIEVERS,
)


@pytest.fixture
def sample_features():
    """7-column feature matrix: [sim, recency, freq, unresolved, utility, weight, goal_rel]"""
    np.random.seed(42)
    n = 20
    features = np.random.rand(n, 7)
    # Make some nodes clearly better
    features[0] = [0.9, 0.1, 0.5, 1.0, 0.5, 0.5, 0.8]  # high sim, old, unresolved
    features[1] = [0.2, 0.9, 0.1, 0.0, 0.5, 0.5, 0.1]  # low sim, recent
    features[2] = [0.8, 0.8, 0.7, 0.0, 0.5, 0.5, 0.9]  # high sim + recent + goal
    return features


@pytest.fixture
def theta_gate():
    """Theta for gate retriever (6 dims: recency, freq, unresolved, utility, weight, goal_rel)"""
    return np.array([0.5, 0.3, 0.2, 0.0, 0.0, 2.5])


@pytest.fixture
def theta_linear():
    """Theta for linear retriever (7 dims: all features)"""
    return np.array([1.0, 0.5, 0.3, 0.2, 0.0, 0.0, 2.5])


class TestRecency:
    def test_returns_recency_column(self, sample_features):
        scores = retrieve_recency(sample_features)
        np.testing.assert_array_equal(scores, sample_features[:, 1])

    def test_most_recent_ranked_higher(self, sample_features):
        scores = retrieve_recency(sample_features)
        # Node 1 has recency=0.9, node 0 has recency=0.1
        assert scores[1] > scores[0]


class TestSimilarity:
    def test_returns_similarity_column(self, sample_features):
        scores = retrieve_similarity(sample_features)
        np.testing.assert_array_equal(scores, sample_features[:, 0])

    def test_most_similar_ranked_higher(self, sample_features):
        scores = retrieve_similarity(sample_features)
        # Node 0 has sim=0.9, node 1 has sim=0.2
        assert scores[0] > scores[1]


class TestGA:
    def test_combines_recency_importance_similarity(self, sample_features):
        scores = retrieve_ga(sample_features)
        # GA = alpha*recency + beta*importance(weight) + gamma*sim
        expected = sample_features[:, 1] + sample_features[:, 5] + sample_features[:, 0]
        np.testing.assert_array_almost_equal(scores, expected)

    def test_node_with_all_high_wins(self, sample_features):
        scores = retrieve_ga(sample_features)
        best_idx = np.argmax(scores)
        # Node 2 has high sim (0.8) + high recency (0.8) + weight (0.5) = 2.1
        assert best_idx == 2


class TestMGAGate:
    def test_gate_structure(self, sample_features, theta_gate):
        """Score = sim × (1 + sigmoid(theta^T f_persistent))"""
        scores = retrieve_mga_gate(sample_features, theta_gate)
        
        sim = sample_features[:, 0]
        f_pers = sample_features[:, 1:]
        expected = sim * (1.0 + expit(f_pers @ theta_gate))
        np.testing.assert_array_almost_equal(scores, expected)

    def test_low_sim_cannot_be_boosted(self, sample_features, theta_gate):
        """Key property: low similarity nodes can't be dragged in by importance."""
        scores = retrieve_mga_gate(sample_features, theta_gate)
        
        # Node 1 has sim=0.2 (low). Even with high recency, gate can't save it.
        # Node 0 has sim=0.9 (high). Gate amplifies it.
        assert scores[0] > scores[1]

    def test_gate_always_positive(self, sample_features, theta_gate):
        """Gate is 1 + sigmoid(...) which is always in [1, 2], so scores >= 0."""
        scores = retrieve_mga_gate(sample_features, theta_gate)
        assert np.all(scores >= 0)

    def test_gate_bounded(self, sample_features, theta_gate):
        """Score bounded by sim * 2 (since sigmoid max = 1)."""
        scores = retrieve_mga_gate(sample_features, theta_gate)
        sim = sample_features[:, 0]
        assert np.all(scores <= sim * 2 + 1e-10)
        assert np.all(scores >= sim * 1 - 1e-10)

    def test_goal_relevance_dominant(self):
        """When theta weights goal_rel heavily, high goal_rel nodes get boosted."""
        n = 10
        features = np.full((n, 7), 0.5)
        features[:, 0] = 0.5  # equal similarity
        features[3, 6] = 1.0  # node 3 has high goal_relevance
        features[7, 6] = 0.0  # node 7 has low goal_relevance
        
        theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.0])  # only goal_rel matters
        scores = retrieve_mga_gate(features, theta)
        
        assert scores[3] > scores[7]


class TestMGALinear:
    def test_linear_combination(self, sample_features, theta_linear):
        scores = retrieve_mga_linear(sample_features, theta_linear)
        expected = sample_features @ theta_linear
        np.testing.assert_array_almost_equal(scores, expected)

    def test_weighted_by_theta(self):
        """Theta determines feature importance."""
        n = 5
        features = np.zeros((n, 7))
        features[0, 6] = 1.0  # only goal_rel
        features[1, 0] = 1.0  # only sim
        
        theta = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0])  # goal_rel weighted 5x sim
        scores = retrieve_mga_linear(features, theta)
        
        assert scores[0] > scores[1]  # goal_rel node wins


class TestTopK:
    def test_returns_k_indices(self):
        scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        indices = top_k(scores, k=3)
        assert len(indices) == 3

    def test_returns_highest_first(self):
        scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        indices = top_k(scores, k=3)
        assert indices[0] == 1  # highest score
        assert indices[1] == 4  # second highest
        assert indices[2] == 2  # third

    def test_k_larger_than_n(self):
        scores = np.array([0.1, 0.9])
        indices = top_k(scores, k=5)
        assert len(indices) == 2

    def test_handles_ties(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        indices = top_k(scores, k=3)
        assert len(indices) == 3


class TestRetrieverRegistry:
    def test_all_retrievers_registered(self):
        assert "recency" in RETRIEVERS
        assert "sim" in RETRIEVERS
        assert "ga" in RETRIEVERS
        assert "mga_gate" in RETRIEVERS
        assert "mga_linear" in RETRIEVERS

    def test_all_retrievers_callable(self):
        for name, fn in RETRIEVERS.items():
            assert callable(fn)
