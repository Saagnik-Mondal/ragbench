import math

import pytest

from ragbench.metrics import mrr_at_k, ndcg_at_k, recall_at_k

REL = {"a": 1, "b": 1}


def test_recall_counts_relevant_in_top_k():
    assert recall_at_k(["a", "x", "b"], REL, 2) == 0.5
    assert recall_at_k(["a", "x", "b"], REL, 3) == 1.0


def test_recall_with_no_relevant_docs_is_zero():
    assert recall_at_k(["a"], {}, 5) == 0.0


def test_mrr_uses_first_relevant_rank():
    assert mrr_at_k(["x", "y", "b"], REL, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y", "z"], REL, 10) == 0.0
    assert mrr_at_k(["x", "y", "b"], REL, 2) == 0.0  # outside the cutoff


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "x"], REL, 10) == pytest.approx(1.0)


def test_ndcg_matches_hand_computation():
    # relevant at ranks 2 and 3: DCG = 1/log2(3) + 1/log2(4); IDCG = 1/log2(2) + 1/log2(3)
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["x", "a", "b"], REL, 10) == pytest.approx(expected)


def test_ndcg_graded_relevance_rewards_putting_best_first():
    graded = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], graded, 5) > ndcg_at_k(["b", "a"], graded, 5)
