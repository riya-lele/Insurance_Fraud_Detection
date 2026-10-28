import numpy as np
import pytest

from src.opt_costs import (bayes_threshold_existing, bayes_threshold_matched, cost_summary,
                           top_k_flags, tune_threshold)


def test_cost_summary_hand_case():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    flag = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    r = cost_summary(y, flag, L=10, c=1)
    assert (r["true_positives"], r["false_positives"], r["missed_fraud"]) == (1, 2, 1)
    assert r["total_cost"] == 13.0 and r["cost_per_claim"] == pytest.approx(1.3)
    assert r["precision"] == pytest.approx(1 / 3) and r["recall"] == pytest.approx(0.5)


def test_trivial_policies():
    y = np.array([1] + [0] * 9)
    assert cost_summary(y, np.zeros(10, bool), 10, 1)["cost_per_claim"] == pytest.approx(1.0)  # L x fraud rate
    assert cost_summary(y, np.ones(10, bool), 10, 1)["cost_per_claim"] == pytest.approx(1.0)   # c per claim


def test_matched_threshold_is_the_cost_minimiser():
    L, c = 10.0, 1.0
    for p in [0.05, 0.09, 0.095, 0.105, 0.2]:
        assert (p > bayes_threshold_matched(L, c)) == (c < p * L)   # flag costs c; skipping costs p*L


def test_existing_threshold_differs_under_this_accounting():
    assert bayes_threshold_existing(10, 1) == pytest.approx(1 / 11)
    assert bayes_threshold_matched(10, 1) == pytest.approx(0.1)


def test_tune_threshold_hand_case():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.3, 0.2])
    # L=10, c=1: top1 = 11/4, top2 = 2/4, top3 = 3/4, top4 = 1, none = 5
    assert tune_threshold(y, s, L=10, c=1) == pytest.approx(0.8)


def test_tune_threshold_infinite_when_flagging_never_pays():
    assert tune_threshold(np.zeros(4), np.array([0.4, 0.3, 0.2, 0.1]), L=10, c=1) == np.inf


def test_top_k_selects_the_right_claims():
    flag = top_k_flags(np.arange(100.0), 0.05)
    assert flag.sum() == 5 and flag[95:].all()