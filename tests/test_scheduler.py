"""Unit tests for scheduler.py (Task 6, 7, 9).

Required checks:
1. Constraints are satisfied when feasible.
2. Solver status is checked and reported.
3. Infeasible cases fall back safely (reduce to minimum rate) instead of crashing.
"""

import logging
import math
from typing import Any

import cvxpy as cp
import numpy as np
import pytest

from ev_sched.hal import EVSEInterface, MeterInterface, SimEVSE, SimMeter
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


def test_mpc_loop_updates_delivered_energy_via_evse_feedback_without_simulator() -> None:
    """Finding 1 verification: Controller tracks battery delivered energy from EVSE telemetry without simulator."""
    from ev_sched.hal import EVSEInterface, MeterInterface

    class HardwareOCPPEVSE(EVSEInterface):
        def __init__(self) -> None:
            self.dispatched: dict[str, float] = {}

        def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
            self.dispatched[ev_id] = power_kw

        def read_session_kw(self, ev_id: str) -> float:
            return self.dispatched.get(ev_id, 0.0)

    class HardwareModbusMeter(MeterInterface):
        def read_feeder_kw(self, t_min: int) -> float:
            return 30.0

        def get_reading_timestamp_s(self, t_min: int) -> float | None:
            return float(t_min * 60.0)

    feeder = Feeder(transformer_limit_kw=60.0, base_load_profile_kw=[20.0] * 4)
    ev = EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=10.0, session_id="EV_HW_TEST")
    tariff = Tariff(price_per_kwh_per_15min=[0.1] * 4)

    evse = HardwareOCPPEVSE()
    meter = HardwareModbusMeter()

    # Pass simulator=None to guarantee logic does not depend on simulator.step()
    res = run_mpc_charging_loop(
        feeder=feeder,
        sessions=[ev],
        tariff=tariff,
        meter=meter,
        evse=evse,
        simulator=None,
        time_step_min=15,
        num_steps=4,
    )

    # In each 15-min step (0.25 h), ev charges at ~10 kW -> 2.5 kWh per step -> 10 kWh total
    assert ev.delivered_energy_kwh > 0.0
    assert np.isclose(ev.delivered_energy_kwh, 10.0, atol=1e-2)
    assert np.isclose(sum(res.total_ev_power_kw) * 0.25, 10.0, atol=1e-2)


def test_mpc_handles_evse_command_exception() -> None:
    """EVSE setpoint dispatch exceptions must not crash MPC and must trigger safe mode."""
    class FailingCommandEVSE(EVSEInterface):
        def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
            raise RuntimeError("EVSE communication timeout during setpoint dispatch")

        def read_session_kw(self, ev_id: str) -> float:
            return 0.0

    class ConstantMeter(MeterInterface):
        def read_feeder_kw(self, t_min: int) -> float:
            return 25.0

        def get_reading_timestamp_s(self, t_min: int) -> float | None:
            return float(t_min * 60.0)

    feeder = Feeder(transformer_limit_kw=60.0, base_load_profile_kw=[20.0] * 4)
    ev = EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.2, session_id="EV_FAIL_CMD")
    tariff = Tariff(price_per_kwh_per_15min=[0.1] * 4)

    evse = FailingCommandEVSE()
    meter = ConstantMeter()

    res = run_mpc_charging_loop(
        feeder=feeder,
        sessions=[ev],
        tariff=tariff,
        meter=meter,
        evse=evse,
        simulator=None,
        time_step_min=15,
        num_steps=4,
    )

    # 1. Controller does not crash and returns full simulation result
    assert res is not None
    assert len(res.total_ev_power_kw) == 4

    # 2. Safe state is entered and reported
    assert any(res.safe_mode_active)
    assert any(res.alarms_raised)

    # 3. Feeder power calculations remain finite and non-negative
    for p in res.total_ev_power_kw:
        assert math.isfinite(p) and p >= 0.0
    for f in res.true_feeder_kw:
        assert math.isfinite(f) and f >= 0.0
    for o in res.overload_kw:
        assert math.isfinite(o) and o >= 0.0


def test_mpc_handles_evse_readback_exception() -> None:
    """EVSE readback exceptions must not crash MPC, must be rejected, and must enter safe mode."""
    class FailingReadbackEVSE(EVSEInterface):
        def __init__(self) -> None:
            self.dispatched: dict[str, float] = {}

        def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
            self.dispatched[ev_id] = power_kw

        def read_session_kw(self, ev_id: str) -> float:
            raise OSError("Submeter bus connection dropped")

    class ConstantMeter(MeterInterface):
        def read_feeder_kw(self, t_min: int) -> float:
            return 25.0

        def get_reading_timestamp_s(self, t_min: int) -> float | None:
            return float(t_min * 60.0)

    feeder = Feeder(transformer_limit_kw=60.0, base_load_profile_kw=[20.0] * 4)
    ev = EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.2, session_id="EV_FAIL_READ")
    tariff = Tariff(price_per_kwh_per_15min=[0.1] * 4)

    evse = FailingReadbackEVSE()
    meter = ConstantMeter()

    res = run_mpc_charging_loop(
        feeder=feeder,
        sessions=[ev],
        tariff=tariff,
        meter=meter,
        evse=evse,
        simulator=None,
        time_step_min=15,
        num_steps=4,
    )

    assert res is not None
    assert any(res.safe_mode_active)
    assert any(res.alarms_raised)

    for p in res.total_ev_power_kw:
        assert math.isfinite(p) and p >= 0.0
    for f in res.true_feeder_kw:
        assert math.isfinite(f) and f >= 0.0
    for o in res.overload_kw:
        assert math.isfinite(o) and o >= 0.0


@pytest.mark.parametrize("invalid_reading", [float("nan"), float("inf"), float("-inf"), -50.0])
def test_mpc_rejects_nan_infinite_and_invalid_evse_readbacks(invalid_reading: float) -> None:
    """EVSE readbacks containing NaN, inf, or negative power must be rejected without poisoning feeder calculations."""
    class InvalidReadbackEVSE(EVSEInterface):
        def __init__(self, val: float) -> None:
            self.val = val

        def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
            pass

        def read_session_kw(self, ev_id: str) -> float:
            return self.val

    class ConstantMeter(MeterInterface):
        def read_feeder_kw(self, t_min: int) -> float:
            return 25.0

        def get_reading_timestamp_s(self, t_min: int) -> float | None:
            return float(t_min * 60.0)

    feeder = Feeder(transformer_limit_kw=60.0, base_load_profile_kw=[20.0] * 4)
    ev = EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.2, session_id="EV_INVALID_READ")
    tariff = Tariff(price_per_kwh_per_15min=[0.1] * 4)

    evse = InvalidReadbackEVSE(invalid_reading)
    meter = ConstantMeter()

    res = run_mpc_charging_loop(
        feeder=feeder,
        sessions=[ev],
        tariff=tariff,
        meter=meter,
        evse=evse,
        simulator=None,
        time_step_min=15,
        num_steps=4,
    )

    assert res is not None

    # Safe state is entered
    assert any(res.safe_mode_active)

    # Invalid reading is rejected: no NaNs or infinities appear in feeder power calculations
    for p in res.total_ev_power_kw:
        assert math.isfinite(p), f"total_ev_power_kw contained non-finite value: {p}"
        assert p >= 0.0
    for f in res.true_feeder_kw:
        assert math.isfinite(f), f"true_feeder_kw contained non-finite value: {f}"
        assert f >= 0.0
    for o in res.overload_kw:
        assert math.isfinite(o), f"overload_kw contained non-finite value: {o}"
        assert o >= 0.0

    # Delivered energy is not poisoned by NaN
    assert math.isfinite(ev.delivered_energy_kwh)


# ==============================================================================
# Audit Regression Tests: Primary Meter Validation & Actuation Emergency Stop
# ==============================================================================

class FaultyPrimaryMeter(MeterInterface):
    """Primary meter simulating communication exceptions, non-finite values, and stale state."""

    def __init__(self, mode: str, is_stale: bool = False, timestamp_offset_min: float = 0.0) -> None:
        self.mode = mode
        self.is_stale = is_stale
        self.timestamp_offset_min = timestamp_offset_min

    def get_reading_timestamp_s(self, t_min: int) -> float | None:
        if self.mode == "none_timestamp":
            return None
        return float((t_min + self.timestamp_offset_min) * 60.0)

    def read_feeder_kw(self, t_min: int) -> float:
        if self.mode == "exception":
            raise RuntimeError("Substation Modbus communication failure")
        if self.mode == "nan":
            return float("nan")
        if self.mode == "inf":
            return float("inf")
        if self.mode == "negative":
            return -25.0
        return 35.0


class DummyFreshEVSE(EVSEInterface):
    """Simple EVSE that tracks setpoints and reflects delivered power."""

    def __init__(self) -> None:
        self.dispatched: dict[str, float] = {}

    def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
        self.dispatched[ev_id] = max(0.0, float(power_kw))

    def read_session_kw(self, ev_id: str) -> float:
        return self.dispatched.get(ev_id, 0.0)


def test_primary_meter_exception_latches_safe_mode_and_applies_fallback() -> None:
    """Primary meter query exceptions must latch safe mode and apply zero-charge or independent cap fallback."""
    feeder = Feeder(transformer_limit_kw=60.0, base_load_profile_kw=[25.0] * 4)
    sessions = [EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.2, session_id="EV_EXC")]
    tariff = Tariff(price_per_kwh_per_15min=[0.1] * 4)

    evse = DummyFreshEVSE()
    meter_exc = FaultyPrimaryMeter("exception")

    # 1. No independent aggregate cap -> fall back to zero charging
    res_zero = run_mpc_charging_loop(
        feeder=feeder,
        sessions=sessions,
        tariff=tariff,
        meter=meter_exc,
        evse=evse,
        simulator=None,
        time_step_min=15,
        num_steps=4,
        max_aggregate_ev_kw=None,
    )
    assert res_zero is not None
    assert all(res_zero.safe_mode_active), "Safe mode was not latched across all steps!"
    assert all(res_zero.alarms_raised), "Alarm was not raised on primary meter failure!"
    assert res_zero.is_protection_active is False
    assert all(p == 0.0 for p in res_zero.total_ev_power_kw), "Zero charging fallback was not enforced!"
    for f in res_zero.true_feeder_kw:
        assert math.isfinite(f) and f >= 0.0
    for o in res_zero.overload_kw:
        assert math.isfinite(o) and o >= 0.0

    # 2. Independent aggregate cap provided -> safe mode latched but capped by independent cap
    res_cap = run_mpc_charging_loop(
        feeder=feeder,
        sessions=sessions,
        tariff=tariff,
        meter=meter_exc,
        evse=evse,
        simulator=None,
        time_step_min=15,
        num_steps=4,
        max_aggregate_ev_kw=5.0,
    )
    assert all(res_cap.safe_mode_active)
    assert all(res_cap.alarms_raised)
    assert res_cap.is_protection_active is True
    assert max(res_cap.total_ev_power_kw) <= 5.0 + 1e-4


@pytest.mark.parametrize("bad_mode", ["nan", "inf", "negative"])
def test_primary_meter_non_finite_and_negative_values(bad_mode: str) -> None:
    """Primary meter returning NaN, inf, or negative values must latch safe mode and apply fallback."""
    feeder = Feeder(transformer_limit_kw=60.0, base_load_profile_kw=[25.0] * 4)
    sessions = [EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.2, session_id="EV_BAD")]
    tariff = Tariff(price_per_kwh_per_15min=[0.1] * 4)

    evse = DummyFreshEVSE()
    meter_bad = FaultyPrimaryMeter(bad_mode)

    res = run_mpc_charging_loop(
        feeder=feeder,
        sessions=sessions,
        tariff=tariff,
        meter=meter_bad,
        evse=evse,
        simulator=None,
        time_step_min=15,
        num_steps=4,
        max_aggregate_ev_kw=None,
    )
    assert all(res.safe_mode_active)
    assert all(res.alarms_raised)
    assert all(p == 0.0 for p in res.total_ev_power_kw)
    for f in res.true_feeder_kw:
        assert math.isfinite(f) and f >= 0.0
    for o in res.overload_kw:
        assert math.isfinite(o) and o >= 0.0


def test_primary_meter_stale_and_unverified_freshness() -> None:
    """Primary meter with is_stale, expired timestamps, or unverified timestamps must latch safe mode."""
    feeder = Feeder(transformer_limit_kw=60.0, base_load_profile_kw=[25.0] * 4)
    sessions = [EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.2, session_id="EV_STALE")]
    tariff = Tariff(price_per_kwh_per_15min=[0.1] * 4)
    evse = DummyFreshEVSE()

    stale_meters = [
        FaultyPrimaryMeter("normal", is_stale=True),
        FaultyPrimaryMeter("normal", timestamp_offset_min=-30.0),  # expired: 30 minutes in the past
        FaultyPrimaryMeter("none_timestamp"),  # unverified freshness
    ]

    for m in stale_meters:
        res = run_mpc_charging_loop(
            feeder=feeder,
            sessions=sessions,
            tariff=tariff,
            meter=m,
            evse=evse,
            simulator=None,
            time_step_min=15,
            num_steps=4,
            max_aggregate_ev_kw=None,
        )
        assert all(res.safe_mode_active), f"Safe mode not latched for stale meter: {m.mode}, {m.is_stale}"
        assert all(res.alarms_raised)
        assert all(p == 0.0 for p in res.total_ev_power_kw)


def test_evse_over_delivery_triggers_actuation_failure_and_emergency_stop() -> None:
    """Significant EVSE over-delivery (> 0.1 kW) must trigger actuation failure and emergency shutdown."""
    class OverDeliveringEVSE(EVSEInterface):
        def __init__(self) -> None:
            self.dispatched: dict[str, float] = {}
            self.emergency_stopped = False

        def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
            self.dispatched[ev_id] = power_kw
            if power_kw == 0.0:
                self.emergency_stopped = True

        def read_session_kw(self, ev_id: str) -> float:
            cmd = self.dispatched.get(ev_id, 0.0)
            if self.emergency_stopped:
                return 0.0
            # Over-deliver: if commanded 2.0 kW, delivers 6.0 kW (+4.0 kW mismatch)
            return cmd + 4.0 if cmd > 0.0 else 0.0

    class ConstantMeter(MeterInterface):
        def read_feeder_kw(self, t_min: int) -> float:
            return 30.0

        def get_reading_timestamp_s(self, t_min: int) -> float | None:
            return float(t_min * 60.0)

    feeder = Feeder(transformer_limit_kw=60.0, base_load_profile_kw=[25.0] * 4)
    sessions = [EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.2, session_id="EV_OVER")]
    tariff = Tariff(price_per_kwh_per_15min=[0.1] * 4)

    evse = OverDeliveringEVSE()
    meter = ConstantMeter()

    res = run_mpc_charging_loop(
        feeder=feeder,
        sessions=sessions,
        tariff=tariff,
        meter=meter,
        evse=evse,
        simulator=None,
        time_step_min=15,
        num_steps=4,
    )

    # 1. Actuation discrepancy was caught
    assert evse.emergency_stopped is True, "Emergency stop set_setpoint_kw(0.0) was not called on mismatch!"
    # 2. Safe mode latched and alarms raised
    assert all(res.safe_mode_active)
    assert any(res.alarms_raised)
    assert res.is_protection_active is False


def test_one_session_command_failure_stops_all_sessions_in_same_step() -> None:
    """An EVSE actuation failure on one session must attempt to stop all sessions in the same step."""
    class ClusterEVSE(EVSEInterface):
        def __init__(self) -> None:
            self.dispatched: dict[str, float] = {}
            self.stopped_sessions: set[str] = set()

        def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
            if ev_id == "EV_FAIL" and power_kw > 0.0:
                raise RuntimeError("EVSE hardware communication fault on EV_FAIL")
            self.dispatched[ev_id] = power_kw
            if power_kw == 0.0:
                self.stopped_sessions.add(ev_id)

        def read_session_kw(self, ev_id: str) -> float:
            if ev_id == "EV_FAIL":
                # Stale or stuck readback on failed EVSE
                return 4.5
            # Healthy EVSE successfully reaches 0.0 when stopped
            if ev_id in self.stopped_sessions:
                return 0.0
            return self.dispatched.get(ev_id, 0.0)

    class ConstantMeter(MeterInterface):
        def read_feeder_kw(self, t_min: int) -> float:
            return 30.0

        def get_reading_timestamp_s(self, t_min: int) -> float | None:
            return float(t_min * 60.0)

    feeder = Feeder(transformer_limit_kw=60.0, base_load_profile_kw=[25.0] * 4)
    sessions = [
        EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.2, session_id="EV_HEALTHY"),
        EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.2, session_id="EV_FAIL"),
    ]
    tariff = Tariff(price_per_kwh_per_15min=[0.1] * 4)

    evse = ClusterEVSE()
    meter = ConstantMeter()

    res = run_mpc_charging_loop(
        feeder=feeder,
        sessions=sessions,
        tariff=tariff,
        meter=meter,
        evse=evse,
        simulator=None,
        time_step_min=15,
        num_steps=4,
    )

    # 1. Emergency stop attempted across ALL sessions
    assert "EV_HEALTHY" in evse.stopped_sessions, "Healthy EV was not stopped when neighbor failed!"

    # 2. Unconfirmed zero power correctly reported for EV_FAIL
    assert any(res.zero_power_unconfirmed), "zero_power_unconfirmed was not flagged!"
    assert any("EV_FAIL" in unconf for unconf in res.unconfirmed_zero_sessions), "EV_FAIL not in unconfirmed list!"
    # EV_HEALTHY reached 0.0 kW, so it should NOT be in unconfirmed list
    for unconf in res.unconfirmed_zero_sessions:
        assert "EV_HEALTHY" not in unconf, "EV_HEALTHY was confirmed zero and should not be unconfirmed!"

    # 3. Safe mode latched
    assert all(res.safe_mode_active)
    assert any(res.alarms_raised)

    for s in sessions:
        assert s.delivered_energy_kwh >= 0.0


