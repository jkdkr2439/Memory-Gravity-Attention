"""
Tests for evaluation metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from metrics import (
    recall_at_k, precision_at_k, ndcg_at_k,
    stale_at_k, noise_at_k, evaluate_retrieval,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a", "b", "c", "d", "e"]
        gold = {"a", "b", "c"}
        assert recall_at_k(retrieved, gold, k=5) == 1.0

    def test_zero_recall(self):
        retrieved = ["x", "y", "z", "w", "v"]
        gold = {"a", "b", "c"}
        assert recall_at_k(retrieved, gold, k=5) == 0.0

    def test_partial_recall(self):
        retrieved = ["a", "x", "b", "y", "z"]
        gold = {"a", "b", "c", "d"}
        assert recall_at_k(retrieved, gold, k=5) == 0.5  # 2/4

    def test_empty_gold(self):
        retrieved = ["a", "b", "c"]
        assert recall_at_k(retrieved, set(), k=5) == 0.0

    def test_k_limits_retrieval(self):
        retrieved = ["x", "y", "z", "a", "b"]  # gold at positions 4,5
        gold = {"a", "b"}
        assert recall_at_k(retrieved, gold, k=3) == 0.0  # only check first 3
        assert recall_at_k(retrieved, gold, k=5) == 1.0  # check all 5


class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = ["a", "b", "c", "d", "e"]
        gold = {"a", "b", "c", "d", "e"}
        assert precision_at_k(retrieved, gold, k=5) == 1.0

    def test_zero_precision(self):
        retrieved = ["x", "y", "z", "w", "v"]
        gold = {"a", "b"}
        assert precision_at_k(retrieved, gold, k=5) == 0.0

    def test_partial_precision(self):
        retrieved = ["a", "x", "b", "y", "z"]
        gold = {"a", "b"}
        assert precision_at_k(retrieved, gold, k=5) == 0.4  # 2/5


class TestNDCGAtK:
    def test_perfect_ndcg(self):
        retrieved = ["a", "b", "c"]
        gold = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, gold, k=3) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        retrieved = ["x", "y", "z"]
        gold = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, gold, k=3) == 0.0

    def test_order_matters(self):
        gold = {"a"}
        # Gold at position 1 vs position 5
        ndcg_first = ndcg_at_k(["a", "x", "y", "z", "w"], gold, k=5)
        ndcg_last = ndcg_at_k(["x", "y", "z", "w", "a"], gold, k=5)
        assert ndcg_first > ndcg_last

    def test_empty_gold(self):
        assert ndcg_at_k(["a", "b"], set(), k=5) == 0.0


class TestStaleAtK:
    def test_no_stale(self):
        retrieved = ["a", "b", "c", "d", "e"]
        stale = {"x", "y"}
        assert stale_at_k(retrieved, stale, k=5) == 0.0

    def test_all_stale(self):
        retrieved = ["a", "b", "c", "d", "e"]
        stale = {"a", "b", "c", "d", "e"}
        assert stale_at_k(retrieved, stale, k=5) == 1.0

    def test_partial_stale(self):
        retrieved = ["a", "b", "c", "d", "e"]
        stale = {"a", "c"}
        assert stale_at_k(retrieved, stale, k=5) == 0.4  # 2/5


class TestNoiseAtK:
    def test_no_noise(self):
        retrieved = ["a", "b", "c", "d", "e"]
        noise = {"x", "y"}
        assert noise_at_k(retrieved, noise, k=5) == 0.0

    def test_some_noise(self):
        retrieved = ["a", "b", "c", "d", "e"]
        noise = {"b", "d"}
        assert noise_at_k(retrieved, noise, k=5) == 0.4


class TestEvaluateRetrieval:
    def test_returns_all_metrics(self):
        result = evaluate_retrieval(
            ["a", "b", "c", "d", "e"],
            gold_ids={"a", "c"},
            stale_ids={"b"},
            noise_ids={"d"},
            k=5,
        )
        assert "recall@k" in result
        assert "precision@k" in result
        assert "ndcg@k" in result
        assert "stale@k" in result
        assert "noise@k" in result
        
        assert result["recall@k"] == 1.0  # 2/2
        assert result["precision@k"] == 0.4  # 2/5
        assert result["stale@k"] == 0.2  # 1/5
        assert result["noise@k"] == 0.2  # 1/5
