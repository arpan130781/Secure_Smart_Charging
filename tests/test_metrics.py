"""Unit tests for metrics.py."""

import numpy as np

from ev_sched.metrics import (
    compute_electricity_cost_usd,
    compute_overload_minutes,
    compute_peak_load_kw,
    compute_run_metrics,
    compute_unmet_energy_kwh,
)
from ev_sched.models import EVSession, Tariff


def test_metrics_computations() -> None:
    # 4 steps (1 hour total, 15 min each = 0.25 h)
    feeder_powers = [80.0, 110.0, 120.0, 90.0]
    ev_powers = [20.0, 30.0, 30.0, 10.0]
    limit_kw = 100.0

    # Overload duration: steps 1 and 2 exceed 100.0 kW -> 2 steps * 15 min = 30 min
    overload_min = compute_overload_minutes(feeder_powers, limit_kw, time_step_min=15)
    assert overload_min == 30.0

    # Peak load: 120.0 kW
    peak = compute_peak_load_kw(feeder_powers)
    assert peak == 120.0

    # Tariff: [0.10, 0.20, 0.20, 0.10]
    # Cost = (0.10 * 20 + 0.20 * 30 + 0.20 * 30 + 0.10 * 10) * 0.25
    #      = (2.0 + 6.0 + 6.0 + 1.0) * 0.25 = 15.0 * 0.25 = 3.75
    tariff = Tariff(price_per_kwh_per_15min=[0.10, 0.20, 0.20, 0.10])
    cost = compute_electricity_cost_usd(ev_powers, tariff, time_step_min=15)
    assert np.isclose(cost, 3.75)

    # Sessions:
    s1 = EVSession(0, 60, energy_req_kwh=20.0, max_rate_kw=11.0, delivered_energy_kwh=18.0)
    s2 = EVSession(0, 60, energy_req_kwh=15.0, max_rate_kw=11.0, delivered_energy_kwh=15.0)
    unmet = compute_unmet_energy_kwh([s1, s2])
    assert np.isclose(unmet, 2.0)

    # RunMetrics container
    rm = compute_run_metrics(feeder_powers, ev_powers, [s1, s2], tariff, limit_kw, 15)
    assert rm.overload_minutes == 30.0
    assert rm.peak_load_kw == 120.0
    assert np.isclose(rm.total_cost_usd, 3.75)
    assert np.isclose(rm.unmet_energy_kwh, 2.0)


def test_energy_conservation_invariant() -> None:
    """Verify that total delivered energy strictly equals the time integral of active EV power."""
    from ev_sched.hal import SimEVSE, SimMeter
    from ev_sched.models import Feeder, Tariff
    from ev_sched.scheduler import run_mpc_charging_loop
    from ev_sched.simulator import (
        ChargingStationSimulator,
        generate_synthetic_sessions,
        run_uncontrolled_charging,
    )

    rng = np.random.default_rng(42)
    cfg = {
        "num_evs": 25,
        "arrival": {"peak_hour": 8.5, "std_hour": 1.5},
        "duration": {"mean_log_min": 5.7, "sigma_log": 0.35},
        "energy": {"mean_kwh": 25.0, "std_kwh": 5.0, "min_kwh": 10.0, "max_kwh": 40.0},
        "max_rate_kw": [7.2, 11.0],
    }
    sessions = generate_synthetic_sessions(cfg, rng)
    feeder = Feeder(transformer_limit_kw=120.0, base_load_profile_kw=[35.0] * 96)
    dt_hours = 15.0 / 60.0

    # 1. Uncontrolled
    res_unc = run_uncontrolled_charging(feeder, sessions, time_step_min=15, num_steps=96)
    sum_e_unc = sum(s.delivered_energy_kwh for s in res_unc.sessions)
    int_p_unc = sum(res_unc.ev_load_kw) * dt_hours
    assert np.isclose(sum_e_unc, int_p_unc, atol=1e-9)

    # 2. MPC
    sessions_mpc = generate_synthetic_sessions(cfg, np.random.default_rng(42))
    sim = ChargingStationSimulator(feeder, sessions_mpc, time_step_min=15, rng=np.random.default_rng(42))
    meter = SimMeter(sim, noise_std_kw=0.3, rng=np.random.default_rng(42))
    evse = SimEVSE(sim)
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 96)
    res_mpc = run_mpc_charging_loop(feeder, sessions_mpc, tariff, meter, evse, simulator=sim, horizon_steps=16)
    sum_e_mpc = sum(s.delivered_energy_kwh for s in res_mpc.sessions)
    int_p_mpc = sum(res_mpc.total_ev_power_kw) * dt_hours
    assert np.isclose(sum_e_mpc, int_p_mpc, atol=1e-9)

