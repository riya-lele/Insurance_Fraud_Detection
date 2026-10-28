import numpy as np

from src.opt_common import k_count, make_splits, topk_metrics


def _labels(n=600, rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, int(n * rate), replace=False)] = 1
    return y


def test_partitions_are_disjoint_and_complete():
    y = _labels()
    folds = list(make_splits(y, 5, 2))
    assert len(folds) == 10
    for sp in folds:
        fit, cal, test = set(sp["fit"]), set(sp["cal"]), set(sp["test"])
        assert not (fit & cal) and not (fit & test) and not (cal & test)
        assert len(fit | cal | test) == len(y)
        ft, fv = set(sp["fit_train"]), set(sp["fitness"])
        assert not (ft & fv)
        assert (ft | fv) == fit                      # fitness split comes only from the fit partition
        assert not (fv & cal) and not (fv & test)    # trials never see calibration or test rows


def test_each_repeat_tests_every_row_once():
    y = _labels()
    folds = list(make_splits(y, 5, 2))
    for r in range(2):
        tests = np.concatenate([folds[r * 5 + i]["test"] for i in range(5)])
        assert sorted(tests) == list(range(len(y)))


def test_splits_are_reproducible():
    y = _labels()
    a = list(make_splits(y, 5, 1))
    b = list(make_splits(y, 5, 1))
    for sa, sb in zip(a, b):
        for key in ("fit", "cal", "test", "fit_train", "fitness"):
            assert np.array_equal(sa[key], sb[key])


def test_fraud_rate_is_preserved():
    y = _labels()
    for sp in make_splits(y, 5, 1):
        assert abs(y[sp["fitness"]].mean() - y.mean()) < 0.03


def test_topk_hand_computed():
    y = np.zeros(100, dtype=int)
    y[[0, 1, 2, 50]] = 1                      # 4 frauds; three are ranked at the very top
    score = 100.0 - np.arange(100)            # descending by position
    m = topk_metrics(y, score)
    assert (m["k_at_1pct"], m["k_at_2pct"], m["k_at_5pct"], m["k_at_10pct"]) == (1, 2, 5, 10)
    assert m["precision_at_1pct"] == 1.0 and m["recall_at_1pct"] == 0.25
    assert m["precision_at_2pct"] == 1.0 and m["recall_at_2pct"] == 0.5
    assert m["precision_at_5pct"] == 0.6 and m["recall_at_5pct"] == 0.75
    assert m["precision_at_10pct"] == 0.3 and m["recall_at_10pct"] == 0.75


def test_k_count_rounding():
    assert k_count(3084, 0.05) == 154
    assert k_count(3084, 0.01) == 31
    assert k_count(10, 0.01) == 1             # never fewer than one claim