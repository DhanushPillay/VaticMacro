import numpy as np

from src.train_model import compute_metrics, naive_baseline_mae


def test_compute_metrics_r2_formula() -> None:
    y_true = np.array([3.0, 3.5, 4.0])
    y_pred = np.array([3.1, 3.4, 4.1])
    m = compute_metrics(y_true, y_pred)
    assert 0 < m["r2"] <= 1
    assert abs(m["mae"] - 0.1) < 1e-9
    assert m["rmse"] > 0


def test_naive_baseline_beats_random() -> None:
    y = np.array([4.0, 4.2, 4.1, 4.3, 4.5])
    mae = naive_baseline_mae(y)
    assert 0 < mae < 1.0
