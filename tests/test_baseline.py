"""Tests for Task 5: Baseline uncontrolled EV charging.

Required checks:
1. Energy delivered never exceeds energy requested.
2. No EV exceeds max_rate_kw at any time step.
3. Uncontrolled charging is deterministic for a fixed seed.
"""

from pathlib import Path

import numpy as np
import yaml

from ev_sched.models import Feeder
from ev_sched.simulator import generate_synthetic_sessions, run_uncontrolled_charging
from experiments.run_baseline import generate_diurnal_base_load


def test_energy_delivered_never_exceeds_energy_requested() -> None:
    config_path = Path("experiments/baseline_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rng = np.random.default_rng(config["seed"])
    sessions = generate_synthetic_sessions(config, rng)

    feeder = Feeder(
        transformer_limit_kw=float(config["feeder"]["transformer_limit_kw"]),
        base_load_profile_kw=[40.0] * 96,
    )

    result = run_uncontrolled_charging(
        feeder=feeder,
        sessions=sessions,
        time_step_min=15,
        num_steps=96,
    )

    for s in result.sessions:
        # Physical law: cannot overcharge past battery target
        assert s.delivered_energy_kwh <= s.energy_req_kwh + 1e-6, (
            f"EV {s.session_id} overcharged: delivered {s.delivered_energy_kwh:.3f} kWh "
            f"> requested {s.energy_req_kwh:.3f} kWh"
        )
        assert s.delivered_energy_kwh >= 0.0


def test_no_ev_exceeds_max_rate_kw() -> None:
    config_path = Path("experiments/baseline_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rng = np.random.default_rng(config["seed"])
    sessions = generate_synthetic_sessions(config, rng)
    feeder = Feeder(
        transformer_limit_kw=float(config["feeder"]["transformer_limit_kw"]),
        base_load_profile_kw=[40.0] * 96,
    )

    result = run_uncontrolled_charging(
        feeder=feeder,
        sessions=sessions,
        time_step_min=15,
        num_steps=96,
    )

    # Verify that in every time step of the simulation, aggregate ev_load_kw does not
    # exceed the sum of max_rate_kw of all vehicles connected at that step
    for step in range(96):
        t_min = step * 15
        connected_evs = [s for s in sessions if s.is_connected(t_min)]
        max_possible_aggregate_kw = sum(s.max_rate_kw for s in connected_evs)
        actual_ev_kw = result.ev_load_kw[step]
        assert actual_ev_kw <= max_possible_aggregate_kw + 1e-6, (
            f"Step {step} (t={t_min} min): actual EV load {actual_ev_kw:.2f} kW "
            f"exceeded maximum connected rating {max_possible_aggregate_kw:.2f} kW"
        )
        # Power per connected vehicle on average cannot exceed maximum individual rating
        if connected_evs:
            max_single_rating = max(s.max_rate_kw for s in connected_evs)
            assert actual_ev_kw <= len(connected_evs) * max_single_rating + 1e-6


def test_uncontrolled_charging_deterministic_for_fixed_seed() -> None:
    config_path = Path("experiments/baseline_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    fixed_seed = 12345

    # Run 1
    rng1 = np.random.default_rng(fixed_seed)
    base_load1 = generate_diurnal_base_load(96, 40.0, 65.0, 25.0, rng1)
    feeder1 = Feeder(120.0, base_load1)
    sessions1 = generate_synthetic_sessions(config, rng1)
    res1 = run_uncontrolled_charging(feeder1, sessions1, 15, 96)

    # Run 2 with identical seed
    rng2 = np.random.default_rng(fixed_seed)
    base_load2 = generate_diurnal_base_load(96, 40.0, 65.0, 25.0, rng2)
    feeder2 = Feeder(120.0, base_load2)
    sessions2 = generate_synthetic_sessions(config, rng2)
    res2 = run_uncontrolled_charging(feeder2, sessions2, 15, 96)

    # Strict reproducibility: outputs must be bit-for-bit identical
    np.testing.assert_array_equal(res1.total_feeder_load_kw, res2.total_feeder_load_kw)
    np.testing.assert_array_equal(res1.ev_load_kw, res2.ev_load_kw)
    np.testing.assert_array_equal(res1.base_load_kw, res2.base_load_kw)
    for s1, s2 in zip(res1.sessions, res2.sessions):
        assert s1.session_id == s2.session_id
        assert s1.arrival_min == s2.arrival_min
        assert s1.departure_min == s2.departure_min
        assert np.isclose(s1.delivered_energy_kwh, s2.delivered_energy_kwh)
