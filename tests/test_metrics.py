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
