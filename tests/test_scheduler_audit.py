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

        def get_reading_timestamp_s(self, t_min: int) -> float | None:
            return float(t_min * 60.0)

        def read_feeder_kw(self, t_min: int) -> float:
            self.read_history.append(t_min)
            return 30.0

    class SpyEVSE(EVSEInterface):
        def __init__(self) -> None:
            self.dispatch_count: dict[str, int] = {}
            self.current_kw: dict[str, float] = {}

        def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
            self.dispatch_count[ev_id] = self.dispatch_count.get(ev_id, 0) + 1
            self.current_kw[ev_id] = power_kw

        def read_session_kw(self, ev_id: str) -> float:
            return self.current_kw.get(ev_id, 0.0)

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

        def get_reading_timestamp_s(self, t_min: int) -> float | None:
            return float(t_min * 60.0)

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


def test_non_aligned_session_window_partial_intervals() -> None:
    """Verify that sessions not aligned to time_step_min correctly model partial intervals.

    Regression case: 10-minute arrival and 20-minute departure on a 15-minute step.
    - Physical dwell is 10 minutes (10 to 20 min).
    - At 7.2 kW, total physical energy capacity is 7.2 * (10 / 60) = 1.2 kWh.
    - Step 0 ([0, 15) min): connected only 10..15 min (5 min, 1/3 interval) -> max power 2.4 kW, energy <= 0.6 kWh.
      Cannot charge before arrival (0..10 min).
    - Step 1 ([15, 30) min): connected only 15..20 min (5 min, 1/3 interval) -> max power 2.4 kW, energy <= 0.6 kWh.
      Cannot count energy after departure (20..30 min).
    - Step 2+ ([30, 60) min): vehicle disconnected -> power == 0.0 kW.
    """
    from ev_sched.hal import SimEVSE, SimMeter
    from ev_sched.scheduler import validate_schedule_constraints
    from ev_sched.simulator import ChargingStationSimulator, run_uncontrolled_charging

    # 1. Optimizer schedule verification
    session = EVSession(
        arrival_min=10,
        departure_min=20,
        energy_req_kwh=5.0,
        max_rate_kw=7.2,
        session_id="EV_PARTIAL",
    )
    feeder = Feeder(transformer_limit_kw=100.0, base_load_profile_kw=[20.0] * 4)
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 4)
    time_step_min = 15
    num_steps = 4
    dt_hours = 15.0 / 60.0  # 0.25 h

    res = solve_charging_schedule(
        sessions=[session],
        feeder=feeder,
        tariff=tariff,
        time_step_min=time_step_min,
        num_steps=num_steps,
    )
    assert res.is_success
    P = res.power_schedule_kw

    # Step 0: [0, 15) min - vehicle arrives at 10 min, connected only for 5 min (1/3 of step)
    # Effective max rate is 7.2 * (5/15) = 2.4 kW. Delivered energy <= 2.4 * 0.25 = 0.6 kWh.
    assert P[0, 0] <= 2.4 + 1e-4, f"Step 0 power {P[0, 0]} exceeds 2.4 kW allowable partial rate"
    energy_step_0 = P[0, 0] * dt_hours
    assert energy_step_0 <= 0.6 + 1e-4, f"Step 0 energy {energy_step_0} exceeds 0.6 kWh (charges before arrival)"

    # Step 1: [15, 30) min - vehicle departs at 20 min, connected only for 5 min (1/3 of step)
    # Effective max rate is 7.2 * (5/15) = 2.4 kW. Delivered energy <= 2.4 * 0.25 = 0.6 kWh.
    assert P[0, 1] <= 2.4 + 1e-4, f"Step 1 power {P[0, 1]} exceeds 2.4 kW allowable partial rate"
    energy_step_1 = P[0, 1] * dt_hours
    assert energy_step_1 <= 0.6 + 1e-4, f"Step 1 energy {energy_step_1} exceeds 0.6 kWh (counts energy after departure)"

    # Steps 2 and 3: [30, 60) min - vehicle already departed
    assert P[0, 2] == pytest.approx(0.0, abs=1e-4)
    assert P[0, 3] == pytest.approx(0.0, abs=1e-4)

    # Total delivered energy across the entire horizon cannot exceed physical limit of 1.2 kWh
    total_energy_delivered = float(np.sum(P[0, :]) * dt_hours)
    assert total_energy_delivered == pytest.approx(1.2, abs=1e-3)
    assert total_energy_delivered <= 7.2 * (10.0 / 60.0) + 1e-4

    # Unmet energy slack captures remaining requested energy (5.0 - 1.2 = 3.8 kWh)
    assert res.unmet_energy_kwh[0] == pytest.approx(3.8, abs=1e-3)

    # 2. Validator rejection of non-compliant candidate schedules
    base_load_arr = np.array(feeder.base_load_profile_kw[:num_steps])
    is_valid, msg = validate_schedule_constraints(
        power_matrix=P,
        unmet_arr=res.unmet_energy_kwh,
        sessions=[session],
        base_load_arr=base_load_arr,
        transformer_limit_kw=feeder.transformer_limit_kw,
        dt_hours=dt_hours,
        num_steps=num_steps,
    )
    assert is_valid, f"Optimal partial schedule rejected by validator: {msg}"

    # Candidate attempting full 7.2 kW in step 0 (charging before arrival) must be rejected
    P_viol_arrival = P.copy()
    P_viol_arrival[0, 0] = 7.2
    is_valid_viol0, msg_viol0 = validate_schedule_constraints(
        power_matrix=P_viol_arrival,
        unmet_arr=res.unmet_energy_kwh,
        sessions=[session],
        base_load_arr=base_load_arr,
        transformer_limit_kw=feeder.transformer_limit_kw,
        dt_hours=dt_hours,
        num_steps=num_steps,
    )
    assert not is_valid_viol0
    assert "exceeds max rate" in msg_viol0

    # Candidate attempting to charge after departure in step 2 must be rejected
    P_viol_dep = P.copy()
    P_viol_dep[0, 2] = 2.0
    is_valid_viol2, msg_viol2 = validate_schedule_constraints(
        power_matrix=P_viol_dep,
        unmet_arr=res.unmet_energy_kwh,
        sessions=[session],
        base_load_arr=base_load_arr,
        transformer_limit_kw=feeder.transformer_limit_kw,
        dt_hours=dt_hours,
        num_steps=num_steps,
    )
    assert not is_valid_viol2
    assert "scheduled outside connection window" in msg_viol2

    # 3. Physical simulator integration (ChargingStationSimulator)
    s_sim = EVSession(10, 20, energy_req_kwh=5.0, max_rate_kw=7.2, session_id="EV_SIM")
    sim = ChargingStationSimulator(feeder=feeder, sessions=[s_sim], time_step_min=time_step_min)

    # Step 0 [0, 15): vehicle arrives at 10 min -> max possible power is 2.4 kW even if commanded 7.2 kW
    sim.set_ev_setpoint_kw("EV_SIM", 7.2)
    p0 = sim.get_session_power_kw("EV_SIM")
    assert p0 == pytest.approx(2.4, abs=1e-4)
    sim.step()
    assert s_sim.delivered_energy_kwh == pytest.approx(0.6, abs=1e-4)

    # Step 1 [15, 30): vehicle departs at 20 min -> max possible power is 2.4 kW even if commanded 7.2 kW
    sim.set_ev_setpoint_kw("EV_SIM", 7.2)
    p1 = sim.get_session_power_kw("EV_SIM")
    assert p1 == pytest.approx(2.4, abs=1e-4)
    sim.step()
    assert s_sim.delivered_energy_kwh == pytest.approx(1.2, abs=1e-4)

    # Step 2 [30, 45): vehicle disconnected -> 0.0 kW delivered
    sim.set_ev_setpoint_kw("EV_SIM", 7.2)
    p2 = sim.get_session_power_kw("EV_SIM")
    assert p2 == pytest.approx(0.0, abs=1e-4)
    sim.step()
    assert s_sim.delivered_energy_kwh == pytest.approx(1.2, abs=1e-4)

    # 4. Uncontrolled charging baseline
    s_unc = EVSession(10, 20, energy_req_kwh=5.0, max_rate_kw=7.2, session_id="EV_UNC")
    res_unc = run_uncontrolled_charging(feeder=feeder, sessions=[s_unc], time_step_min=time_step_min, num_steps=num_steps)
    assert res_unc.sessions[0].delivered_energy_kwh == pytest.approx(1.2, abs=1e-4)
    assert res_unc.ev_load_kw[0] == pytest.approx(2.4, abs=1e-4)
    assert res_unc.ev_load_kw[1] == pytest.approx(2.4, abs=1e-4)
    assert res_unc.ev_load_kw[2] == pytest.approx(0.0, abs=1e-4)
    assert res_unc.ev_load_kw[3] == pytest.approx(0.0, abs=1e-4)

    # 5. Full closed-loop MPC simulation
    s_mpc = EVSession(10, 20, energy_req_kwh=5.0, max_rate_kw=7.2, session_id="EV_MPC")
    sim_mpc = ChargingStationSimulator(feeder=feeder, sessions=[s_mpc], time_step_min=time_step_min)
    meter_mpc = SimMeter(simulator=sim_mpc)
    evse_mpc = SimEVSE(simulator=sim_mpc)

    res_mpc = run_mpc_charging_loop(
        feeder=feeder,
        sessions=[s_mpc],
        tariff=tariff,
        meter=meter_mpc,
        evse=evse_mpc,
        simulator=sim_mpc,
        num_steps=num_steps,
        time_step_min=time_step_min,
        horizon_steps=num_steps,
    )
    assert res_mpc.total_ev_power_kw[0] == pytest.approx(2.4, abs=1e-3)
    assert res_mpc.total_ev_power_kw[1] == pytest.approx(2.4, abs=1e-3)
    assert res_mpc.total_ev_power_kw[2] == pytest.approx(0.0, abs=1e-3)
    assert res_mpc.total_ev_power_kw[3] == pytest.approx(0.0, abs=1e-3)
    assert s_mpc.delivered_energy_kwh == pytest.approx(1.2, abs=1e-3)
