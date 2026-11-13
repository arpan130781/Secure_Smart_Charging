"""Unit tests for the Embedded C99 Core and Pure Control Classical Detectors."""

import subprocess
from pathlib import Path

import numpy as np

from ev_sched.c_bridge import LIB_AVAILABLE, c_schedule_water_filling
from ev_sched.detector import KalmanChiSquareDetector
from ev_sched.models import EVSession
from ev_sched.scheduler import solve_water_filling_step


def test_kalman_chisq_detector_normal_and_attack() -> None:
    detector = KalmanChiSquareDetector(q_process=0.5, r_meas=0.09, chisq_threshold=9.0)
    rng = np.random.default_rng(42)

    # 40 steps of clean telemetry
    for _ in range(40):
        noise = rng.normal(0.0, 0.2)
        alert = detector.update(reported_feeder_kw=50.0 + noise, expected_feeder_kw=50.0)
        assert not alert.is_alarm, "Normal sensor noise should not trigger Chi-Sq alarm"

    # Sudden 25 kW under-reporting attack
    alert_attack = detector.update(reported_feeder_kw=25.0, expected_feeder_kw=50.0)
    assert alert_attack.is_alarm, "25 kW jump anomaly must trigger Chi-Sq alarm"
    assert alert_attack.statistic_value > 9.0


def test_water_filling_step_respects_headroom_and_caps() -> None:
    sessions = [
        EVSession(session_id="EV1", arrival_min=0, departure_min=120, energy_req_kwh=10.0, max_rate_kw=22.0),
        EVSession(session_id="EV2", arrival_min=0, departure_min=60, energy_req_kwh=20.0, max_rate_kw=22.0),
        EVSession(session_id="EV3", arrival_min=0, departure_min=240, energy_req_kwh=5.0, max_rate_kw=22.0),
    ]

    # Available headroom = 100 - 60 - 5 = 35 kW
    allocations = solve_water_filling_step(
        transformer_limit_kw=100.0,
        base_load_kw=60.0,
        safety_margin_kw=5.0,
        sessions=sessions,
        current_time_min=0,
        dt_hours=0.25,
    )

    total_allocated = sum(allocations.values())
    assert total_allocated <= 35.0001, f"Allocated power ({total_allocated}) exceeded headroom (35 kW)"
    assert total_allocated > 30.0, "Allocated power under-utilized available headroom"

    # EV2 has shortest dwell (60 min) with highest remaining energy (20 kWh), should get highest power
    assert allocations["EV2"] >= allocations["EV1"]
    assert allocations["EV2"] >= allocations["EV3"]

    # Test fail-safe latching
    safe_allocations = solve_water_filling_step(
        transformer_limit_kw=100.0,
        base_load_kw=60.0,
        safety_margin_kw=5.0,
        sessions=sessions,
        current_time_min=0,
        dt_hours=0.25,
        fail_safe_active=True,
        safe_cap_kw=5.0,
    )
    assert sum(safe_allocations.values()) <= 5.0001


def test_c_bridge_library_invocation() -> None:
    assert LIB_AVAILABLE, "libseccharge shared library must be compiled and loadable"

    sessions_data = [
        {"max_rate_kw": 22.0, "remaining_energy_kwh": 20.0, "dwell_hours": 2.0},
        {"max_rate_kw": 22.0, "remaining_energy_kwh": 10.0, "dwell_hours": 1.0},
    ]

    tot_kw, per_ev = c_schedule_water_filling(
        transformer_limit_kw=100.0,
        base_load_kw=50.0,
        safety_margin_kw=5.0,
        time_step_hours=0.25,
        sessions_data=sessions_data,
        fail_safe_active=False,
    )

    assert tot_kw <= 45.0001
    assert len(per_ev) == 2
    assert per_ev[0] >= 0.0 and per_ev[1] >= 0.0


def test_core_c_standalone_make_test() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    core_c_dir = repo_root / "core_c"

    res = subprocess.run(["make", "test"], check=False, cwd=core_c_dir, capture_output=True, text=True)
    assert res.returncode == 0, f"make test failed:\n{res.stderr}\n{res.stdout}"
    assert "assertions passed" in res.stdout
