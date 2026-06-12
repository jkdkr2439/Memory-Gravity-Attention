"""
Tests for the attention-entropy diagnostic (lock detection).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from entropy_diagnostic import (
    score_entropy, normalized_entropy, max_entropy,
    top_k_concentration, diversity_at_k,
    detect_lock, compare_retrievers_entropy,
)


class TestScoreEntropy:
    def test_uniform_scores_high_entropy(self):
        """Uniform scores should have maximum entropy."""
        scores = np.ones(100)
        h = score_entropy(scores)
        # Should be close to log(100)
        assert h == pytest.approx(np.log(100), rel=0.01)

    def test_peaked_scores_low_entropy(self):
        """One dominant score should have low entropy."""
        scores = np.zeros(100)
        scores[0] = 100.0  # one very high score
        h = score_entropy(scores)
        assert h < 1.0  # much less than log(100) ≈ 4.6

    def test_entropy_non_negative(self):
        """Entropy is always >= 0."""
        for _ in range(10):
            scores = np.random.rand(50)
            assert score_entropy(scores) >= 0

    def test_temperature_effect(self):
        """Lower temperature should give lower entropy (sharper)."""
        scores = np.array([1.0, 0.8, 0.5, 0.3, 0.1])
        h_low_t = score_entropy(scores, temperature=0.1)
        h_high_t = score_entropy(scores, temperature=10.0)
        assert h_low_t < h_high_t


class TestNormalizedEntropy:
    def test_uniform_gives_one(self):
        scores = np.ones(50)
        assert normalized_entropy(scores) == pytest.approx(1.0, rel=0.01)

    def test_peaked_gives_near_zero(self):
        scores = np.zeros(50)
        scores[0] = 1000.0
        h = normalized_entropy(scores)
        assert h < 0.1

    def test_bounded_01(self):
        for _ in range(20):
            scores = np.random.rand(30) * 10
            h = normalized_entropy(scores)
            assert 0 <= h <= 1.0 + 1e-6

    def test_single_element(self):
        assert normalized_entropy(np.array([5.0])) == 0.0


class TestMaxEntropy:
    def test_log_n(self):
        assert max_entropy(100) == pytest.approx(np.log(100))
        assert max_entropy(1) == 0.0


class TestTopKConcentration:
    def test_uniform_low_concentration(self):
        """Uniform scores: top-k gets k/n of the mass."""
        scores = np.ones(100)
        conc = top_k_concentration(scores, k=5)
        assert conc == pytest.approx(0.05, rel=0.01)

    def test_peaked_high_concentration(self):
        """One dominant score: top-k gets most mass."""
        scores = np.zeros(100)
        scores[0] = 100.0
        conc = top_k_concentration(scores, k=5)
        assert conc > 0.9

    def test_bounded(self):
        scores = np.random.rand(50)
        conc = top_k_concentration(scores, k=5)
        assert 0 < conc <= 1.0


class TestDiversityAtK:
    def test_identical_embeddings_one_cluster(self):
        """If all top-k embeddings are identical, diversity = 1."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.1])
        # All embeddings identical
        embeddings = np.ones((7, 64))
        div = diversity_at_k(scores, embeddings, k=5, cluster_threshold=0.85)
        assert div == 1

    def test_orthogonal_embeddings_max_diversity(self):
        """If top-k embeddings are orthogonal, diversity = k."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        # Create orthogonal embeddings
        embeddings = np.eye(5)  # each is a unit vector in different direction
        div = diversity_at_k(scores, embeddings, k=5, cluster_threshold=0.85)
        assert div == 5

    def test_mixed_clusters(self):
        """Some similar, some different → intermediate diversity."""
        scores = np.array([0.9, 0.85, 0.8, 0.75, 0.7])
        embeddings = np.zeros((5, 10))
        embeddings[0, 0] = 1.0  # cluster A
        embeddings[1, 0] = 0.95; embeddings[1, 1] = 0.05  # cluster A (similar to 0)
        embeddings[2, 5] = 1.0  # cluster B
        embeddings[3, 5] = 0.9; embeddings[3, 6] = 0.1  # cluster B (similar to 2)
        embeddings[4, 9] = 1.0  # cluster C
        # Normalize
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        div = diversity_at_k(scores, embeddings, k=5, cluster_threshold=0.85)
        # Should be ~3 clusters (A, B, C)
        assert 2 <= div <= 4


class TestDetectLock:
    def test_locked_pattern(self):
        """Consistently low entropy → LOCKED."""
        entropies = [0.1, 0.12, 0.08, 0.11, 0.09, 0.13, 0.1, 0.07, 0.11, 0.1]
        result = detect_lock(entropies, threshold=0.3)
        assert result["status"] == "LOCKED"
        assert result["lock_ratio"] == 1.0

    def test_healthy_pattern(self):
        """High, variable entropy → HEALTHY or NORMAL (no lock)."""
        entropies = [0.75, 0.82, 0.6, 0.9, 0.71, 0.85, 0.68, 0.92, 0.77, 0.8]
        result = detect_lock(entropies, threshold=0.3)
        assert result["status"] in ("HEALTHY", "NORMAL")
        assert result["lock_ratio"] == 0.0

    def test_partial_lock(self):
        """Mix of low and high → PARTIAL_LOCK."""
        entropies = [0.1, 0.8, 0.15, 0.7, 0.2, 0.1, 0.75, 0.12, 0.8, 0.1]
        result = detect_lock(entropies, threshold=0.3)
        assert result["status"] in ("PARTIAL_LOCK", "NORMAL")
        assert 0 < result["lock_ratio"] < 1.0

    def test_returns_all_fields(self):
        entropies = [0.5] * 10
        result = detect_lock(entropies)
        assert "status" in result
        assert "description" in result
        assert "mean_entropy" in result
        assert "std_entropy" in result
        assert "lock_ratio" in result
        assert "n_queries" in result


class TestCompareRetrieversEntropy:
    def test_identifies_locked_retriever(self):
        """Should flag retriever with consistently low entropy."""
        retriever_scores = {
            "locked_one": [np.array([100.0] + [0.01] * 99) for _ in range(10)],
            "healthy_one": [np.random.rand(100) for _ in range(10)],
        }
        result = compare_retrievers_entropy(retriever_scores, n_nodes=100)
        
        assert "locked_one" in result["summary"]["locked"]
        assert "healthy_one" in result["summary"]["healthy"]

    def test_returns_per_retriever_results(self):
        retriever_scores = {
            "a": [np.random.rand(50) for _ in range(5)],
            "b": [np.random.rand(50) for _ in range(5)],
        }
        result = compare_retrievers_entropy(retriever_scores, n_nodes=50)
        assert "a" in result["per_retriever"]
        assert "b" in result["per_retriever"]
        assert "summary" in result
