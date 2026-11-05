"""Unit tests verifying mathematical invariants, fuzz cases, and MPC telemetry isolation."""

import numpy as np
import pytest

from ev_sched.hal import EVSEInterface, MeterInterface
from ev_sched.models import EVSession, Feeder, Tariff
from ev_sched.scheduler import run_mpc_charging_loop, solve_charging_schedule


def test_numerical_solution_constraints_strictly_satisfied() -> None:
    """Verify that returned numerical values satisfy bounds, window, and feeder capacity."""
    dt_hours = 15.0 / 60.0
    tol = 1e-4

    sessions = [
        EVSession(0, 180, energy_req_kwh=20.0, max_rate_kw=11.0, session_id="EV_1"),
        EVSession(60, 240, energy_req_kwh=15.0, max_rate_kw=7.2, session_id="EV_2"),
        EVSession(120, 300, energy_req_kwh=25.0, max_rate_kw=11.0, session_id="EV_3"),
    ]
    base_load = [20.0] * 24
    transformer_limit = 35.0  # Tight capacity
    feeder = Feeder(transformer_limit_kw=transformer_limit, base_load_profile_kw=base_load)
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 24)

    res = solve_charging_schedule(sessions, feeder, tariff, time_step_min=15, num_steps=24)
    assert res.is_success
    P = res.power_schedule_kw

    # 1. Non-negativity
    assert np.all(P >= -tol)

    # 2. Connection window and max rate
    for i, s in enumerate(sessions):
        arr_step = s.arrival_min // 15
        dep_step = int(np.ceil(s.departure_min / 15))
        assert np.all(P[i, :arr_step] <= tol)
        assert np.all(P[i, dep_step:] <= tol)
        assert np.all(P[i, arr_step:dep_step] <= s.max_rate_kw + tol)

        # No overcharging
        delivered = float(np.sum(P[i, arr_step:dep_step]) * dt_hours)
        assert delivered <= s.energy_req_kwh + tol

    # 3. Transformer limit in every step
    total_load = np.sum(P, axis=0) + np.array(base_load)
    assert np.all(total_load <= transformer_limit + tol)


def test_receding_horizon_queries_meter_and_applies_step_zero_only() -> None:
    """Verify receding horizon reads meter at each step and dispatches only column 0."""
    class SpyMeter(MeterInterface):
        def __init__(self) -> None:
            self.read_history: list[int] = []

        def read_feeder_kw(self, t_min: int) -> float:
            self.read_history.append(t_min)
            return 30.0

    class SpyEVSE(EVSEInterface):
        def __init__(self) -> None:
            self.dispatch_count: dict[str, int] = {}

        def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
            self.dispatch_count[ev_id] = self.dispatch_count.get(ev_id, 0) + 1

        def read_session_kw(self, ev_id: str) -> float:
            return 0.0

    meter = SpyMeter()
    evse = SpyEVSE()
    sessions = [EVSession(0, 120, energy_req_kwh=10.0, max_rate_kw=7.2, session_id="EV_1")]
    feeder = Feeder(transformer_limit_kw=50.0, base_load_profile_kw=[25.0] * 8)
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 8)

    _ = run_mpc_charging_loop(
        feeder=feeder,
        sessions=sessions,
        tariff=tariff,
        meter=meter,
        evse=evse,
        num_steps=8,
        time_step_min=15,
        horizon_steps=4,
    )

    assert len(meter.read_history) == 8
    assert meter.read_history == [step * 15 for step in range(8)]
    assert evse.dispatch_count["EV_1"] == 8


def test_telemetry_isolation_against_direct_simulator_calls() -> None:
    """Verify that MPC controller never reads from simulator during optimization."""
    class TrapSimulator:
        def __init__(self) -> None:
            self.current_time_min = 0

        def get_feeder_power_kw(self, t_min: int) -> float:
            return 50.0

        def get_total_ev_power_kw(self) -> float:
            raise RuntimeError("Simulator direct access attempted!")

        def step(self) -> None:
            self.current_time_min += 15

    class MockMeter(MeterInterface):
        def read_feeder_kw(self, t_min: int) -> float:
            return 25.0

    class MockEVSE(EVSEInterface):
        def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
            pass

        def read_session_kw(self, ev_id: str) -> float:
            return 0.0

    sessions = [EVSession(0, 60, energy_req_kwh=5.0, max_rate_kw=7.2, session_id="EV_1")]
    feeder = Feeder(transformer_limit_kw=50.0, base_load_profile_kw=[25.0] * 4)
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 4)

    # Must complete without calling get_total_ev_power_kw on TrapSimulator
    res = run_mpc_charging_loop(
        feeder=feeder,
        sessions=sessions,
        tariff=tariff,
        meter=MockMeter(),
        evse=MockEVSE(),
        simulator=TrapSimulator(),
        num_steps=4,
    )
    assert res.measured_feeder_kw == [25.0] * 4
    assert res.true_feeder_kw == [50.0] * 4


@pytest.mark.parametrize(
    "name,sessions,limit,base",
    [
        ("empty_fleet", [], 100.0, 30.0),
        ("zero_energy_ev", [EVSession(0, 120, energy_req_kwh=0.0, max_rate_kw=11.0, session_id="E1")], 100.0, 30.0),
        ("single_slot_stay", [EVSession(30, 45, energy_req_kwh=5.0, max_rate_kw=11.0, session_id="E2")], 100.0, 30.0),
        ("simultaneous_arrivals", [EVSession(0, 180, energy_req_kwh=10.0, max_rate_kw=11.0, session_id=f"E{i}") for i in range(8)], 100.0, 30.0),
        ("infeasible_base_exceeds_limit", [EVSession(0, 180, energy_req_kwh=10.0, max_rate_kw=11.0, session_id="E_FAIL")], 40.0, 60.0),
        ("massive_over_demand", [EVSession(0, 180, energy_req_kwh=50.0, max_rate_kw=50.0, session_id=f"E{i}") for i in range(10)], 50.0, 10.0),
    ],
)
def test_fuzz_edge_cases(name: str, sessions: list[EVSession], limit: float, base: float) -> None:
    """Fuzz testing extreme and corner cases."""
    feeder = Feeder(transformer_limit_kw=limit, base_load_profile_kw=[base] * 12)
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 12)

    res = solve_charging_schedule(sessions, feeder, tariff, num_steps=12)

    if base > limit:
        assert not res.is_success
        assert np.all(res.power_schedule_kw == 0.0)
    else:
        assert res.is_success
        P = res.power_schedule_kw
        assert np.all(P >= -1e-6)
        if sessions:
            total_load = np.sum(P, axis=0) + base
            assert np.all(total_load <= limit + 1e-4)
