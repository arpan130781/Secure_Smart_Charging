"""Unit tests for scheduler.py (Task 6, 7, 9).

Required checks:
1. Constraints are satisfied when feasible.
2. Solver status is checked and reported.
3. Infeasible cases fall back safely (reduce to minimum rate) instead of crashing.
"""

import logging
from typing import Any

import cvxpy as cp
import numpy as np
import pytest

from ev_sched.hal import SimEVSE, SimMeter
from ev_sched.models import EVSession, Feeder, Tariff
from ev_sched.scheduler import (
    run_mpc_charging_loop,
    solve_charging_schedule,
)
from ev_sched.simulator import ChargingStationSimulator


def test_cvxpy_constraints_satisfied_when_feasible() -> None:
    # 2 EVs, 4 time steps (1 hour total, dt = 15 min = 0.25 h)
    # Feeder limit 50 kW, base load 20 kW -> available headroom 30 kW
    feeder = Feeder(
        transformer_limit_kw=50.0,
        base_load_profile_kw=[20.0, 20.0, 20.0, 20.0],
    )
    # EV1: requests 5.0 kWh, max 11 kW, connected all 4 steps
    ev1 = EVSession(
        arrival_min=0,
        departure_min=60,
        energy_req_kwh=5.0,
        max_rate_kw=11.0,
        session_id="EV_01",
    )
    # EV2: requests 5.0 kWh, max 11 kW, connected all 4 steps
    ev2 = EVSession(
        arrival_min=0,
        departure_min=60,
        energy_req_kwh=5.0,
        max_rate_kw=11.0,
        session_id="EV_02",
    )
    tariff = Tariff(price_per_kwh_per_15min=[0.20, 0.10, 0.10, 0.20])

    res = solve_charging_schedule(
        sessions=[ev1, ev2],
        feeder=feeder,
        tariff=tariff,
        time_step_min=15,
        num_steps=4,
    )

    # 1. Check solver status
    assert res.is_success
    assert res.status.lower() in ["optimal", "optimal_inaccurate"]

    P = res.power_schedule_kw
    dt_hours = 0.25

    # 2. Check per-EV max rate
    assert np.all(P[0, :] <= 11.0 + 1e-4)
    assert np.all(P[1, :] <= 11.0 + 1e-4)
    assert np.all(P >= 0.0)

    # 3. Check transformer limit constraint at every step
    total_ev = np.sum(P, axis=0)
    total_feeder = np.array(feeder.base_load_profile_kw) + total_ev
    assert np.all(total_feeder <= 50.0 + 1e-4)

    # 4. Check energy requested delivered
    e1_delivered = float(np.sum(P[0, :]) * dt_hours)
    e2_delivered = float(np.sum(P[1, :]) * dt_hours)
    assert np.isclose(e1_delivered, 5.0, atol=1e-3)
    assert np.isclose(e2_delivered, 5.0, atol=1e-3)
    assert np.all(res.unmet_energy_kwh <= 1e-3)

    # 5. Check tariff optimization: solver should prefer cheaper middle slots (0.10 $/kWh)
    # Slots 1 and 2 are cheaper than slots 0 and 3
    assert P[0, 1] + P[0, 2] >= P[0, 0] + P[0, 3]


def test_solver_status_checked_and_reported() -> None:
    feeder = Feeder(transformer_limit_kw=100.0, base_load_profile_kw=[40.0] * 8)
    ev = EVSession(
        arrival_min=0,
        departure_min=120,
        energy_req_kwh=10.0,
        max_rate_kw=11.0,
        session_id="EV_STATUS_TEST",
    )
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 8)

    res = solve_charging_schedule(
        sessions=[ev],
        feeder=feeder,
        tariff=tariff,
        time_step_min=15,
        num_steps=8,
    )

    assert hasattr(res, "status")
    assert isinstance(res.status, str)
    assert res.status.lower() in ["optimal", "optimal_inaccurate"]


def test_infeasible_case_safe_fallback() -> None:
    # Intentionally impossible case: Base load is 60 kW, but transformer limit is only 30 kW
    # No EV charging is safe; solver cannot satisfy feeder limit with positive headroom
    feeder = Feeder(
        transformer_limit_kw=30.0,
        base_load_profile_kw=[60.0] * 4,
    )
    ev = EVSession(
        arrival_min=0,
        departure_min=60,
        energy_req_kwh=15.0,
        max_rate_kw=11.0,
        session_id="EV_FALLBACK",
    )
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 4)

    # Must NOT raise exception / crash
    res = solve_charging_schedule(
        sessions=[ev],
        feeder=feeder,
        tariff=tariff,
        time_step_min=15,
        num_steps=4,
    )

    # Verifies solver status is reported
    assert res.status is not None

    # Verifies safe fallback: power setpoints reduced to minimum rate (0.0 kW)
    assert np.all(res.power_schedule_kw == 0.0)
    # Unmet energy is flagged appropriately
    assert np.isclose(res.unmet_energy_kwh[0], 15.0, atol=1e-3)


def test_receding_horizon_mpc_loop_execution() -> None:
    feeder = Feeder(
        transformer_limit_kw=80.0,
        base_load_profile_kw=[40.0] * 8,
    )
    ev1 = EVSession(
        arrival_min=0,
        departure_min=120,
        energy_req_kwh=10.0,
        max_rate_kw=11.0,
        session_id="EV_01",
    )
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 8)

    sim = ChargingStationSimulator(feeder=feeder, sessions=[ev1], time_step_min=15)
    meter = SimMeter(sim)
    evse = SimEVSE(sim)

    mpc_res = run_mpc_charging_loop(
        feeder=feeder,
        sessions=[ev1],
        tariff=tariff,
        meter=meter,
        evse=evse,
        simulator=sim,
        time_step_min=15,
        num_steps=8,
    )

    # Verification: feeder limit strictly respected
    assert np.all(np.array(mpc_res.overload_kw) == 0.0)
    assert len(mpc_res.solver_statuses) == 8
    # Delivered energy matches target
    assert np.isclose(ev1.delivered_energy_kwh, 10.0, atol=1e-3)


def test_mpc_loop_mitigates_attack_when_detection_enabled() -> None:
    """Task 15: Verify that when detector raises an alarm, safe mode eliminates overload."""
    from ev_sched.attacks import ConstantBiasMeter
    from ev_sched.detector import CUSUMDetector

    # Feeder: 50 kW limit, 35 kW base load -> nominal headroom 15 kW
    feeder = Feeder(transformer_limit_kw=50.0, base_load_profile_kw=[35.0] * 8)
    ev1 = EVSession(
        arrival_min=0,
        departure_min=120,
        energy_req_kwh=10.0,
        max_rate_kw=25.0,
        session_id="EV_ATK_MITIGATION",
    )
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 8)

    sim = ChargingStationSimulator(feeder=feeder, sessions=[ev1], time_step_min=15)
    base_meter = SimMeter(sim, noise_std_kw=0.0)
    # Under-reporting attack: subtracts 25 kW between t=30 and t=90 min
    attacked_meter = ConstantBiasMeter(
        inner_meter=base_meter,
        start_min=30,
        duration_min=60,
        delta_kw=25.0,
    )
    evse = SimEVSE(sim)
    detector = CUSUMDetector(threshold=18.0, slack_kw=1.5, instant_threshold_kw=20.0)

    res_mitigated = run_mpc_charging_loop(
        feeder=feeder,
        sessions=[ev1],
        tariff=tariff,
        meter=attacked_meter,
        evse=evse,
        simulator=sim,
        time_step_min=15,
        num_steps=8,
        detector=detector,
        conservative_margin_kw=5.0,
    )

    # 1. Verify alarm was raised and safe mode was activated
    assert any(res_mitigated.alarms_raised)
    assert any(res_mitigated.safe_mode_active)

    # 2. Verify that under safe mode, true physical overload is completely prevented!
    assert np.all(np.array(res_mitigated.overload_kw) == 0.0)


def test_monkeypatch_solver_error_falls_back_to_second_solver(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Requirement 6(a): Monkeypatch problem.solve to raise SolverError for first solver and check second is used."""
    feeder = Feeder(transformer_limit_kw=50.0, base_load_profile_kw=[20.0] * 4)
    ev = EVSession(arrival_min=0, departure_min=60, energy_req_kwh=5.0, max_rate_kw=11.0, session_id="EV_FALLBACK_TEST")
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 4)

    real_solve = cp.Problem.solve
    solvers_called = []

    def mock_solve(self: cp.Problem, *args: Any, **kwargs: Any) -> Any:
        current_solver = kwargs.get("solver")
        solvers_called.append(current_solver)
        # Fail the first solver called with SolverError
        if len(solvers_called) == 1:
            raise cp.error.SolverError(f"Simulated solver crash for {current_solver}")
        return real_solve(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", mock_solve)

    with caplog.at_level(logging.WARNING):
        res = solve_charging_schedule(
            sessions=[ev],
            feeder=feeder,
            tariff=tariff,
            time_step_min=15,
            num_steps=4,
        )

    assert res.is_success
    assert len(solvers_called) >= 2
    assert res.solver_used is not None
    assert res.solver_used == solvers_called[1]
    assert "failed with SolverError" in caplog.text


def test_non_solver_exception_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 6(b): Raise a non-solver exception and check it propagates."""
    feeder = Feeder(transformer_limit_kw=50.0, base_load_profile_kw=[20.0] * 4)
    ev = EVSession(arrival_min=0, departure_min=60, energy_req_kwh=5.0, max_rate_kw=11.0, session_id="EV_ERR_PROP")
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 4)

    def mock_solve_raise_value_error(self: cp.Problem, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("Invalid dimension or user parameter error")

    monkeypatch.setattr(cp.Problem, "solve", mock_solve_raise_value_error)

    with pytest.raises(ValueError, match="Invalid dimension or user parameter error"):
        solve_charging_schedule(
            sessions=[ev],
            feeder=feeder,
            tariff=tariff,
            time_step_min=15,
            num_steps=4,
        )


def test_infeasible_problem_checks_fallback_and_logged_status(caplog: pytest.LogCaptureFixture) -> None:
    """Requirement 6(c): Infeasible case (transformer limit below base load) checks fallback and logged status."""
    # Transformer limit is 30 kW, but unmanaged base load is 60 kW
    feeder = Feeder(transformer_limit_kw=30.0, base_load_profile_kw=[60.0] * 4)
    ev = EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=11.0, session_id="EV_INFEAS_LOG")
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 4)

    with caplog.at_level(logging.WARNING):
        res = solve_charging_schedule(
            sessions=[ev],
            feeder=feeder,
            tariff=tariff,
            time_step_min=15,
            num_steps=4,
        )

    # Infeasibility triggers fallback: no power scheduled, unmet energy recorded
    assert not res.is_success
    assert "infeasible" in res.status.lower()
    assert np.all(res.power_schedule_kw == 0.0)
    assert np.isclose(res.unmet_energy_kwh[0], 10.0, atol=1e-3)

    # Verify that the infeasible non-optimal status was logged using logging
    assert any("infeasible" in record.message.lower() for record in caplog.records)


def test_solver_used_is_recorded_and_explicit_solver_never_substituted() -> None:
    """Requirement 6(d): Check that solver_used is recorded and explicit solver is never substituted silently."""
    feeder = Feeder(transformer_limit_kw=50.0, base_load_profile_kw=[20.0] * 4)
    ev = EVSession(arrival_min=0, departure_min=60, energy_req_kwh=5.0, max_rate_kw=11.0, session_id="EV_SOLVER_REC")
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 4)

    # 1. Default solve records solver_used
    res_default = solve_charging_schedule([ev], feeder, tariff, time_step_min=15, num_steps=4)
    assert res_default.is_success
    assert res_default.solver_used is not None
    assert isinstance(res_default.solver_used, str)
    assert res_default.is_inaccurate is False

    # 2. Explicit solver passed is recorded and strictly used
    res_osqp = solve_charging_schedule([ev], feeder, tariff, time_step_min=15, num_steps=4, solver="OSQP")
    assert res_osqp.is_success
    assert res_osqp.solver_used == "OSQP"

    # 3. Explicit solver that fails does not substitute another silently
    res_invalid = solve_charging_schedule([ev], feeder, tariff, time_step_min=15, num_steps=4, solver="NONEXISTENT_SOLVER")
    assert not res_invalid.is_success
    assert res_invalid.solver_used == "NONEXISTENT_SOLVER"


