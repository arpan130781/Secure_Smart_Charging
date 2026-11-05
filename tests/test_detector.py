"""Unit tests for detector.py (Task 12, Task 13)."""

import numpy as np
import pytest
from ev_sched.detector import CUSUMDetector, IsolationForestDetector


def test_cusum_detector_normal_operation() -> None:
    detector = CUSUMDetector(threshold=18.0, slack_kw=1.5, instant_threshold_kw=20.0)
    rng = np.random.default_rng(42)

    # 50 normal steps with small noise
    for _ in range(50):
        noise = rng.normal(0.0, 0.4)
        expected = 60.0
        reported = expected + noise
        alert = detector.update(reported_feeder_kw=reported, expected_feeder_kw=expected)
        assert not alert.is_alarm
        assert alert.statistic_value < 5.0


def test_cusum_detector_detects_under_reporting_bias() -> None:
    detector = CUSUMDetector(threshold=18.0, slack_kw=1.5, instant_threshold_kw=20.0)

    # Step 0: Normal
    alert0 = detector.update(reported_feeder_kw=70.0, expected_feeder_kw=70.0)
    assert not alert0.is_alarm

    # Step 1: Attack begins (30 kW under-reporting bias)
    # residual = 40.0 - 70.0 = -30.0 kW -> exceeds instant_threshold (20.0 kW)
    alert1 = detector.update(reported_feeder_kw=40.0, expected_feeder_kw=70.0)
    assert alert1.is_alarm
    assert alert1.residual_kw == -30.0


def test_cusum_detector_detects_ramp_attack() -> None:
    detector = CUSUMDetector(threshold=18.0, slack_kw=1.5, instant_threshold_kw=25.0)

    # Slowly increasing bias: 2 kW, 4 kW, 6 kW, 8 kW, 10 kW...
    alarm_triggered = False
    for step in range(1, 15):
        bias = step * 2.0  # 2 kW drift per step
        reported = 70.0 - bias
        alert = detector.update(reported_feeder_kw=reported, expected_feeder_kw=70.0)
        if alert.is_alarm:
            alarm_triggered = True
            break

    assert alarm_triggered


def test_isolation_forest_detector_train_and_predict() -> None:
    rng = np.random.default_rng(123)
    detector = IsolationForestDetector(contamination=0.01, random_state=42)

    # Generate synthetic clean training data (500 steps)
    # residual ~ N(0, 0.4^2), delta_r ~ N(0, 0.5^2), ratio ~ 1.0 + N(0, 0.01^2)
    residuals = rng.normal(0.0, 0.4, 500)
    delta_r = rng.normal(0.0, 0.5, 500)
    ratios = 1.0 + rng.normal(0.0, 0.01, 500)
    X_clean = np.column_stack([residuals, delta_r, ratios])

    detector.fit_from_attack_free_runs(X_clean)
    assert detector.is_fitted

    # Test on a normal point
    alert_norm = detector.update(reported_feeder_kw=65.2, expected_feeder_kw=65.0)
    assert not alert_norm.is_alarm

    # Test on an attacked point (e.g. 35 kW under-reporting)
    alert_atk = detector.update(reported_feeder_kw=30.0, expected_feeder_kw=65.0)
    assert alert_atk.is_alarm
    assert alert_atk.residual_kw == -35.0
