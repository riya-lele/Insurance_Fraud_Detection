import pytest
from src.score import zone_for


def test_zone_moves_with_L():
    assert zone_for(0.20, L=1.0, c=1.0) == "approve"      # threshold 0.5
    assert zone_for(0.20, L=10.0, c=1.0) == "investigate"  # threshold 0.091


def test_smoke_score_one_row():
    from src.final_model import load_final, DEFAULT_PATH
    from src.data import load_dataset
    from src.score import score_claim
    if not DEFAULT_PATH.exists():
        pytest.skip("model not trained")
    bundle = load_final()
    X, y, _, _ = load_dataset(verbose=False)
    r = score_claim(bundle, dict(X.iloc[0]))
    assert 0.0 <= r["prob"] <= 1.0 and r["zone"] in {"approve", "review", "investigate"}