"""Unit tests for hal.py (MeterInterface, EVSEInterface, SimMeter, SimEVSE)."""

import pytest

from ev_sched.hal import EVSEInterface, MeterInterface, SimEVSE, SimMeter
from ev_sched.models import EVSession, Feeder
from ev_sched.simulator import ChargingStationSimulator


def test_abstract_interfaces_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        MeterInterface()  # type: ignore

    with pytest.raises(TypeError):
        EVSEInterface()  # type: ignore


def test_sim_meter_and_sim_evse_interaction() -> None:
    feeder = Feeder(
        transformer_limit_kw=100.0,
        base_load_profile_kw=[35.0] * 96,
    )
    s1 = EVSession(
        arrival_min=0,
        departure_min=120,
        energy_req_kwh=20.0,
        max_rate_kw=11.0,
        session_id="EV_01",
    )
    sim = ChargingStationSimulator(feeder=feeder, sessions=[s1], time_step_min=15)
    meter: MeterInterface = SimMeter(sim)
    evse: EVSEInterface = SimEVSE(sim)

    # Initial state: setpoint is 0.0 -> EV power is 0.0
    assert meter.read_feeder_kw(t_min=0) == 35.0
    assert evse.read_session_kw("EV_01") == 0.0

    # Dispatch setpoint of 8.0 kW to EV_01 via EVSEInterface
    evse.set_setpoint_kw("EV_01", 8.0)
    assert evse.read_session_kw("EV_01") == 8.0

    # Total feeder power measured by MeterInterface is base (35) + EV (8) = 43.0 kW
    assert meter.read_feeder_kw(t_min=0) == 43.0

    # Dispatch setpoint higher than EV max_rate_kw (e.g. 25.0 kW) -> clamped to 11.0 kW
    evse.set_setpoint_kw("EV_01", 25.0)
    assert evse.read_session_kw("EV_01") == 11.0
    assert meter.read_feeder_kw(t_min=0) == 46.0  # 35 + 11

    # Step simulator (15 min = 0.25 h -> 11 kW * 0.25 h = 2.75 kWh delivered)
    sim.step()
    assert s1.delivered_energy_kwh == 2.75


def test_sim_meter_idempotency_and_rng_isolation() -> None:
    feeder = Feeder(transformer_limit_kw=100.0, base_load_profile_kw=[35.0] * 4)
    sim = ChargingStationSimulator(feeder=feeder, sessions=[], time_step_min=15)
    meter = SimMeter(sim, noise_std_kw=0.5, seed=100)

    # 1. Idempotency: reading the same timestamp multiple times returns identical values
    val1 = meter.read_feeder_kw(t_min=0)
    val2 = meter.read_feeder_kw(t_min=0)
    assert val1 == val2

    # 2. Simulator RNG state is not perturbed by meter reads
    sim_rng_state_before = sim.rng.bit_generator.state
    _ = meter.read_feeder_kw(t_min=15)
    sim_rng_state_after = sim.rng.bit_generator.state
    assert sim_rng_state_before == sim_rng_state_after
