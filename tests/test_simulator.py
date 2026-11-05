"""Unit tests for simulator.py (synthetic session generation & uncontrolled charging)."""

from pathlib import Path

import numpy as np
import yaml

from ev_sched.models import EVSession, Feeder
from ev_sched.simulator import generate_synthetic_sessions, run_uncontrolled_charging


def test_synthetic_session_generation() -> None:
    config_path = Path("experiments/baseline_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rng = np.random.default_rng(config["seed"])
    sessions = generate_synthetic_sessions(config, rng)

    # Check fleet size (must be 20-50 EVs)
    assert 20 <= len(sessions) <= 50
    assert len(sessions) == 30

    for s in sessions:
        # 24-hour day bounds: 0 to 1440 min
        assert 0 <= s.arrival_min < 1440
        assert s.arrival_min < s.departure_min <= 1440
        assert 12.0 <= s.energy_req_kwh <= 55.0
        assert s.max_rate_kw in [7.2, 11.0]
        assert s.delivered_energy_kwh == 0.0


def test_uncontrolled_charging_simulation() -> None:
    feeder = Feeder(
        transformer_limit_kw=100.0,
        base_load_profile_kw=[30.0] * 96,
    )
    s1 = EVSession(
        arrival_min=0,
        departure_min=120,   # 2 hours
        energy_req_kwh=10.0,
        max_rate_kw=10.0,
        session_id="EV_01",
    )
    res = run_uncontrolled_charging(feeder, [s1], time_step_min=15, num_steps=8)

    assert len(res.step_times_min) == 8
    assert len(res.total_feeder_load_kw) == 8
    # Step 0: 10 kW * 0.25 h = 2.5 kWh delivered -> total feeder = 30 + 10 = 40 kW
    # Step 1: 10 kW * 0.25 h = 2.5 kWh delivered -> total 5.0 kWh
    # Step 2: 10 kW * 0.25 h = 2.5 kWh delivered -> total 7.5 kWh
    # Step 3: 10 kW * 0.25 h = 2.5 kWh delivered -> total 10.0 kWh (complete!)
    # Step 4: EV done -> 0 kW -> total feeder = 30 kW
    assert np.isclose(res.ev_load_kw[0], 10.0)
    assert np.isclose(res.ev_load_kw[1], 10.0)
    assert np.isclose(res.ev_load_kw[2], 10.0)
    assert np.isclose(res.ev_load_kw[3], 10.0)
    assert np.isclose(res.ev_load_kw[4], 0.0)
    assert np.isclose(s1.delivered_energy_kwh, 10.0)
