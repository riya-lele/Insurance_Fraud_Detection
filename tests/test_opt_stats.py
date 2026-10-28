import numpy as np
import pandas as pd
import pytest

from src.opt_stats import holm, nadeau_bengio, paired_from_frame


def test_nadeau_bengio_matches_hand_formula():
    a = np.array([0.30, 0.32, 0.29, 0.31, 0.33])
    b = np.array([0.28, 0.30, 0.30, 0.29, 0.30])
    r = nadeau_bengio(a, b, n_train=9000, n_test=3000)
    d = a - b
    se = np.sqrt((1 / 5 + 3000 / 9000) * d.var(ddof=1))
    assert r["mean_diff"] == pytest.approx(d.mean())
    assert r["se_corrected"] == pytest.approx(se)
    assert r["t"] == pytest.approx(d.mean() / se)
    assert r["a_higher_in"] == 4


def test_correction_widens_the_interval():
    a = np.linspace(0.30, 0.34, 15)
    b = a - 0.01 + np.random.default_rng(0).normal(0, 0.005, 15)
    corrected = nadeau_bengio(a, b, 9000, 3000)
    naive = nadeau_bengio(a, b, 1e12, 1)               # ratio ~ 0 gives the uncorrected test
    assert corrected["p_raw"] > naive["p_raw"]


def test_pairs_are_aligned_by_fold_id():
    df = pd.DataFrame({"fold": [0, 1, 2, 0, 1], "model": ["A"] * 3 + ["B"] * 2,
                       "m": [1.0, 2.0, 3.0, 0.5, 1.5]})
    a, b, folds = paired_from_frame(df, "m", "A", "B")
    assert list(folds) == [0, 1] and list(a) == [1.0, 2.0] and list(b) == [0.5, 1.5]


def test_too_few_pairs_raise():
    with pytest.raises(ValueError):
        nadeau_bengio([1.0, 2.0], [0.5, 1.0], 9000, 3000)


def test_holm_hand_case():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_ignores_nan():
    out = holm([0.01, np.nan, 0.02])
    assert out[0] == pytest.approx(0.02) and out[2] == pytest.approx(0.02) and np.isnan(out[1])