
from src.mp216.monitor import calculate_drift_status


def test_stable_feature():
    assert calculate_drift_status(100, 105) == "OK"


def test_large_increase():
    assert calculate_drift_status(5, 10) == "DRIFT ALERT"


def test_large_decrease():
    assert calculate_drift_status(100, 50) == "DRIFT ALERT"


def test_zero_baseline():
    assert calculate_drift_status(0, 10) == "INSUFFICIENT DATA"


def test_missing_value():
    assert calculate_drift_status(
        float("nan"), 10
    ) == "INSUFFICIENT DATA"
