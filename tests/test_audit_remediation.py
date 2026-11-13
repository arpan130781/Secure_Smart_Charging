"""Focused tests verifying the 10 audit and remediation fixes.

Tests:
1. Detector FAR clean-day denominator matches clean intervals actually evaluated.
2. Session ID resolution: unique non-empty IDs resolved for simulator storage, EVSE commands, and telemetry reads.
3. Schedule dispatch hardening: OPTIMAL_INACCURATE results violating constraints are rejected and fall back safely.
4. Independent protective constraint prevents physical transformer overload during adaptive attack.
5. CAN security: rejects extended 29-bit CAN IDs as invalid frame format.
6. CAN security: rejects payload lengths != 8 bytes as malformed.
7. CAN security: detects nominal-timing replay attacks using authenticated freshness (counter / MAC).
8. CAN security: fails safely (0.0 A current setpoint) on invalid or anomalous frames.
9. Headline experiment: configurable transformer limit and safe-mode margin passed to plot.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import can
import cvxpy as cp
import numpy as np
import pytest

from can_security.bms_protocol import (
    CAN_ID_BMS_STATUS,
    encode_bms_status,
    encode_bms_status_canfd,
)
from can_security.can_attacks import inject_nominal_timing_replay_attack
from can_security.can_detector import CANSecurityDetector
from ev_sched.hal import EVSEInterface, MeterInterface, SimEVSE, SimMeter
from ev_sched.models import EVSession, Feeder, Tariff, resolve_session_ids
from ev_sched.scheduler import (
    run_mpc_charging_loop,
    solve_charging_schedule,
    validate_schedule_constraints,
)
from ev_sched.simulator import ChargingStationSimulator
from experiments.run_adaptive_attacker_limitation import (
    run_adaptive_attack_limitation_experiment,
)
from experiments.run_attack_and_detection_experiments import (
    compute_detection_summary_metrics,
)
from experiments.run_headline_experiment import plot_headline_figure


# ==============================================================================
# Issue 1: Detector FAR clean-day denominator matches clean intervals
# ==============================================================================
def test_detector_far_clean_day_denominator_matches_evaluated_clean_intervals():
    # 192 intervals total, 168 clean intervals (False), 24 attack intervals (True)
    y_true = [False] * 168 + [True] * 24
    y_pred = [False] * 168 + [True] * 24
    # Introduce 2 false alarms in the clean portion
    y_pred[10] = True
    y_pred[20] = True

    data = {
        "y_true": y_true,
        "y_pred": y_pred,
        "latencies": [0.0],
        "episode_outcomes": [{"seed": 1, "detected": True}],
    }

    # When total_clean_days is omitted, it must calculate clean_days = 168 / 96 = 1.75 days
    metrics = compute_detection_summary_metrics(data)
    expected_clean_days = 168 / 96.0  # 1.75 days
    expected_far = 2 / expected_clean_days  # 1.14 alarms/day
    assert pytest.approx(metrics["false_alarms_per_day"], 0.01) == round(expected_far, 2)


# ==============================================================================
# Issue 2: Session ID Handling (unique, non-empty, matching HAL & Sim)
# ==============================================================================
def test_session_id_resolution_empty_whitespace_and_duplicates():
    # Create sessions with missing, empty, whitespace, and duplicate IDs
    s1 = EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.0, session_id="")
    s2 = EVSession(arrival_min=15, departure_min=90, energy_req_kwh=15.0, max_rate_kw=11.0, session_id="   ")
    s3 = EVSession(arrival_min=30, departure_min=120, energy_req_kwh=8.0, max_rate_kw=7.0, session_id="EV_DUP")
    s4 = EVSession(arrival_min=45, departure_min=180, energy_req_kwh=12.0, max_rate_kw=7.0, session_id="EV_DUP")

    sessions = [s1, s2, s3, s4]
    resolved = resolve_session_ids(sessions)

    # All resolved IDs must be non-empty strings
    for s in resolved:
        assert isinstance(s.session_id, str)
        assert len(s.session_id.strip()) > 0

    # All resolved IDs must be strictly unique
    ids = [s.session_id for s in resolved]
    assert len(ids) == len(set(ids))

    # Test simulator integration: simulator keys match session.session_id exactly
    feeder = Feeder(transformer_limit_kw=100.0, base_load_profile_kw=[20.0] * 96)
    sim = ChargingStationSimulator(feeder=feeder, sessions=sessions)
    for s in sessions:
        assert s.session_id in sim.sessions
        assert sim.sessions[s.session_id] is s

    # Test EVSEInterface setpoint and read work through the resolved session_id
    evse = SimEVSE(sim)
    evse.set_setpoint_kw(s1.session_id, 7.0)
    assert sim.setpoints_kw[s1.session_id] == 7.0
    actual_kw = evse.read_session_kw(s1.session_id)
    assert actual_kw == 7.0


# ==============================================================================
# Issue 3: Harden Schedule Dispatch on OPTIMAL_INACCURATE
# ==============================================================================
def test_optimal_inaccurate_violating_constraints_falls_back_safely():
    s = EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.0, session_id="EV_1")
    feeder = Feeder(transformer_limit_kw=50.0, base_load_profile_kw=[30.0] * 4)
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 4)

    # Mock cp.Problem.solve to return OPTIMAL_INACCURATE with violating power matrix (e.g. 500 kW exceeding max rate)
    with patch("cvxpy.Problem.solve") as mock_solve, patch("cvxpy.Problem.status", new="optimal_inaccurate"):
        mock_solve.return_value = None

        # Create schedule solve attempt
        res = solve_charging_schedule(
            sessions=[s],
            feeder=feeder,
            tariff=tariff,
            time_step_min=15,
            num_steps=4,
        )

        # Inaccurate solution that violates max rate must be rejected
        # and fallback schedule must be dispatched
        assert res.is_success is False
        assert res.status == "optimal_inaccurate_invalid"
        assert res.is_inaccurate is True
        # Fallback schedule commands 0 power
        assert np.all(res.power_schedule_kw == 0.0)


def test_validate_schedule_constraints_catches_all_violations():
    s = EVSession(arrival_min=15, departure_min=45, energy_req_kwh=5.0, max_rate_kw=10.0, session_id="EV_1")
    base_load = np.array([20.0, 20.0, 20.0, 20.0])
    limit = 50.0
    dt_hours = 0.25

    # 1. Negative power
    p_neg = np.array([[-1.0, 5.0, 5.0, 0.0]])
    is_valid, msg = validate_schedule_constraints(p_neg, np.zeros(1), [s], base_load, limit, dt_hours)
    assert not is_valid and "Negative" in msg

    # 2. Max rate violation
    p_rate = np.array([[0.0, 15.0, 5.0, 0.0]])
    is_valid, msg = validate_schedule_constraints(p_rate, np.zeros(1), [s], base_load, limit, dt_hours)
    assert not is_valid and "max rate" in msg

    # 3. Power outside connection window (step 0: t=0-15 min, arrival is 15 min)
    p_window = np.array([[5.0, 5.0, 5.0, 0.0]])
    is_valid, msg = validate_schedule_constraints(p_window, np.zeros(1), [s], base_load, limit, dt_hours)
    assert not is_valid and "connection window" in msg

    # 4. Energy overcharged (10 kW * 0.25h * 2 = 5 kWh > 3 kWh target, inside window steps 1 and 2)
    s_energy = EVSession(arrival_min=15, departure_min=60, energy_req_kwh=3.0, max_rate_kw=10.0, session_id="EV_1")
    p_energy = np.array([[0.0, 10.0, 10.0, 0.0]])
    is_valid, msg = validate_schedule_constraints(p_energy, np.zeros(1), [s_energy], base_load, limit, dt_hours)
    assert not is_valid and "energy delivered" in msg

    # 5. Feeder limit exceeded (base 20 + EV 35 = 55 > 50 kW limit, inside window step 1)
    s_feeder = EVSession(arrival_min=15, departure_min=60, energy_req_kwh=10.0, max_rate_kw=40.0, session_id="EV_1")
    p_feeder = np.array([[0.0, 35.0, 0.0, 0.0]])
    u_feeder = np.array([1.25])  # Delivered 8.75 + 1.25 = 10.0 target satisfies slack balance
    is_valid, msg = validate_schedule_constraints(p_feeder, u_feeder, [s_feeder], base_load, limit, dt_hours)
    assert not is_valid and "Transformer capacity violated" in msg

    # 6. Negative unmet energy slack
    u_neg = np.array([-0.5])
    is_valid, msg = validate_schedule_constraints(p_feeder, u_neg, [s_feeder], base_load, limit, dt_hours)
    assert not is_valid and "Negative unmet energy slack" in msg

    # 7. Unmet energy balance violated (delivered 2.5 + slack 0.0 < target 10.0 kWh)
    p_under = np.array([[0.0, 5.0, 5.0, 0.0]])
    is_valid, msg = validate_schedule_constraints(p_under, np.zeros(1), [s_feeder], base_load, limit, dt_hours)
    assert not is_valid and "unmet energy balance violated" in msg


# ==============================================================================
# Issue 4: Independent Protective Constraint in Adaptive Attack
# ==============================================================================
def test_independent_protective_constraint_prevents_adaptive_overload():
    # Run unprotected: adaptive attacker evades CUSUM and causes 120 min overload up to 90 kW
    unprotected = run_adaptive_attack_limitation_experiment(seed=42)
    assert unprotected["peak_true_load_kw"] > unprotected["transformer_limit_kw"]
    assert unprotected["overload_minutes"] > 0.0
    assert unprotected["alarms_raised"] == 0  # CUSUM completely blind!

    # Run protected with independent cluster constraint max_aggregate_ev_kw = 5.0 kW
    protected = run_adaptive_attack_limitation_experiment(seed=42, max_aggregate_ev_kw=5.0)
    assert protected["peak_true_load_kw"] <= protected["transformer_limit_kw"]
    assert protected["overload_minutes"] == 0.0


# ==============================================================================
# Issue 5 & 6: CAN Security Claims, Frame Validation & Freshness
# ==============================================================================
def test_can_security_extended_id_rejected():
    detector = CANSecurityDetector()
    msg = encode_bms_status(
        current_demand_a=100.0,
        voltage_limit_v=400.0,
        soc_percent=50.0,
        pack_temp_c=25.0,
        is_extended_id=True,  # Malformed: standard 11-bit required
    )
    alert = detector.evaluate_message(msg)
    assert alert.is_alarm
    assert alert.alarm_type == "invalid_frame_format"
    assert alert.safe_current_setpoint_a == 0.0


def test_can_security_exact_payload_length_enforced():
    detector = CANSecurityDetector()
    # 7-byte truncated frame
    msg_short = can.Message(
        arbitration_id=CAN_ID_BMS_STATUS,
        data=b"\x00\x00\x00\x00\x00\x00\x00",
        is_extended_id=False,
        timestamp=1.0,
    )
    alert = detector.evaluate_message(msg_short)
    assert alert.is_alarm
    assert alert.alarm_type == "payload_malformed"
    assert alert.safe_current_setpoint_a == 0.0

    # 9-byte frame
    msg_long = can.Message(
        arbitration_id=CAN_ID_BMS_STATUS,
        data=b"\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        is_extended_id=False,
        timestamp=2.0,
    )
    alert = detector.evaluate_message(msg_long)
    assert alert.is_alarm
    assert alert.alarm_type == "payload_malformed"
    assert alert.safe_current_setpoint_a == 0.0


def test_can_security_authenticated_freshness_detects_nominal_timing_replay():
    auth_key = b"super_secret_evse_key_32bytes_!"
    detector = CANSecurityDetector(enforce_freshness=True, auth_key=auth_key)

    # Record 3 frames with sequential counters (0, 1, 2)
    buffer = [
        encode_bms_status(
            current_demand_a=80.0,
            voltage_limit_v=380.0,
            soc_percent=50.0,
            pack_temp_c=25.0,
            timestamp=1.0 + i * 0.100,
            counter=i,
            auth_key=auth_key,
        )
        for i in range(3)
    ]

    # Process legitimate frames
    for msg in buffer:
        alert = detector.evaluate_message(msg)
        assert not alert.is_alarm
        assert alert.safe_current_setpoint_a == 80.0

    # Attacker replays frame 0 at timestamp 1.300 s (exact nominal 100 ms period!)
    # Timing detector sees Delta t = 100 ms and passes timing.
    replayed_msgs = inject_nominal_timing_replay_attack(
        [buffer[0]], replay_start_time=1.300, period_s=0.100
    )
    alert_replay = detector.evaluate_message(replayed_msgs[0])

    # Authenticated freshness catches stale counter (counter=0 <= last_counter=2)
    assert alert_replay.is_alarm
    assert alert_replay.alarm_type == "freshness_anomaly"
    assert alert_replay.safe_current_setpoint_a == 0.0  # Fails safely to 0 A!


def test_can_security_mac_verification_catches_forged_payload():
    auth_key = b"secret_key"
    detector = CANSecurityDetector(auth_key=auth_key)

    msg = encode_bms_status(
        current_demand_a=100.0,
        voltage_limit_v=380.0,
        soc_percent=50.0,
        pack_temp_c=25.0,
        auth_key=auth_key,
    )

    # Attacker modifies current in bytes 0-1 without knowledge of auth_key
    tampered_data = bytearray(msg.data)
    tampered_data[0] = 0xFF
    tampered_msg = can.Message(
        arbitration_id=msg.arbitration_id,
        data=bytes(tampered_data),
        is_extended_id=False,
        timestamp=1.0,
    )

    alert = detector.evaluate_message(tampered_msg)
    assert alert.is_alarm
    assert alert.alarm_type == "auth_failure"
    assert alert.safe_current_setpoint_a == 0.0


# ==============================================================================
# Issue 7: Configured Transformer Limits Passed to Plotting Code
# ==============================================================================
def test_headline_configured_limits_passed_to_plot(tmp_path: Path):
    time_series = {
        "normal": [50.0] * 96,
        "under_attack": [110.0] * 96,
        "under_attack_with_detection": [70.0] * 96,
    }
    png_path = tmp_path / "test_headline.png"

    # Must accept arbitrary transformer limit and safe margin without crashing or hardcoding
    plot_headline_figure(
        time_series_data=time_series,
        transformer_limit_kw=95.0,
        safe_margin_kw=12.0,
        output_png_path=png_path,
    )
    assert png_path.exists()


# ==============================================================================
# Audit Follow-up 1: CAN Authentication & Missing Key Rejection
# ==============================================================================
def test_can_security_missing_key_rejected_in_secure_mode():
    with pytest.raises(ValueError, match="auth_key"):
        CANSecurityDetector(secure_mode=True, auth_key=b"")

    with pytest.raises(ValueError, match="auth_key"):
        CANSecurityDetector(enforce_auth=True, auth_key=b"")


# ==============================================================================
# Audit Follow-up 2: CAN Replay Protection Across Detector Restarts & Wraparound
# ==============================================================================
def test_can_security_replay_rejected_across_detector_reset():
    auth_key = b"secret_session_key_32bytes_1234"
    detector = CANSecurityDetector(secure_mode=True, auth_key=auth_key)

    # Frame generated in initial epoch (boot_epoch=0)
    msg_epoch0 = encode_bms_status(
        current_demand_a=70.0,
        voltage_limit_v=380.0,
        soc_percent=50.0,
        pack_temp_c=25.0,
        counter=1,
        auth_key=auth_key,
        epoch=0,
        timestamp=1.0,
    )
    alert = detector.evaluate_message(msg_epoch0)
    assert not alert.is_alarm

    # Detector restarts / resets -> boot_epoch advances to 1
    detector.reset()

    # Adversary replays msg_epoch0 as the first frame after reset at next expected time
    msg_replayed = can.Message(
        arbitration_id=msg_epoch0.arbitration_id,
        data=msg_epoch0.data,
        is_extended_id=msg_epoch0.is_extended_id,
        timestamp=1.1,
    )
    alert_replayed = detector.evaluate_message(msg_replayed)
    assert alert_replayed.is_alarm
    assert alert_replayed.alarm_type == "auth_failure"
    assert alert_replayed.safe_current_setpoint_a == 0.0

    # Legitimate frame generated with updated epoch (epoch=1) passes
    msg_epoch1 = encode_bms_status(
        current_demand_a=75.0,
        voltage_limit_v=380.0,
        soc_percent=51.0,
        pack_temp_c=25.0,
        counter=2,
        auth_key=auth_key,
        epoch=1,
        timestamp=1.2,
    )
    alert_valid = detector.evaluate_message(msg_epoch1)
    assert not alert_valid.is_alarm
    assert alert_valid.safe_current_setpoint_a == 75.0


def test_can_security_counter_wraparound():
    auth_key = b"secret_session_key_32bytes_1234"
    detector = CANSecurityDetector(secure_mode=True, auth_key=auth_key)

    # Frame at counter=254
    msg254 = encode_bms_status(80.0, 380.0, 50.0, 25.0, counter=254, auth_key=auth_key, timestamp=1.0)
    assert not detector.evaluate_message(msg254).is_alarm

    # Frame at counter=255
    msg255 = encode_bms_status(80.0, 380.0, 50.0, 25.0, counter=255, auth_key=auth_key, timestamp=1.1)
    assert not detector.evaluate_message(msg255).is_alarm

    # Valid wraparound to counter=2 (jump of 3 <= max_counter_jump=16)
    msg2 = encode_bms_status(80.0, 380.0, 50.0, 25.0, counter=2, auth_key=auth_key, timestamp=1.2)
    alert_wrap = detector.evaluate_message(msg2)
    assert not alert_wrap.is_alarm

    # Excessive jump during wraparound: from 2 to 30 (jump of 28 > 16)
    msg30 = encode_bms_status(80.0, 380.0, 50.0, 25.0, counter=30, auth_key=auth_key, timestamp=1.3)
    alert_jump = detector.evaluate_message(msg30)
    assert alert_jump.is_alarm
    assert alert_jump.alarm_type == "freshness_anomaly"


# ==============================================================================
# Audit Follow-up 3: Real Fail-Safe Actuation Connected to EVSE
# ==============================================================================
def test_can_security_actuates_failsafe_setpoint_on_evse():
    feeder = Feeder(transformer_limit_kw=100.0, base_load_profile_kw=[20.0] * 10)
    session = EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=22.0, session_id="EV_ACT")
    sim = ChargingStationSimulator(feeder=feeder, sessions=[session])
    evse = SimEVSE(sim)

    # Initially setpoint is 22 kW
    evse.set_setpoint_kw("EV_ACT", 22.0)
    assert evse.read_session_kw("EV_ACT") == 22.0

    # Connect detector to EVSE with target session "EV_ACT"
    detector = CANSecurityDetector(evse=evse, ev_id="EV_ACT")

    # Tampered message arrives (out of bounds current 500 A)
    bad_msg = encode_bms_status(current_demand_a=500.0, voltage_limit_v=380.0, soc_percent=50.0, pack_temp_c=25.0)
    alert = detector.evaluate_message(bad_msg)
    assert alert.is_alarm

    # EVSE setpoint must be actively driven to 0.0 kW
    assert evse.read_session_kw("EV_ACT") == 0.0
    assert sim.setpoints_kw["EV_ACT"] == 0.0


# ==============================================================================
# Audit Follow-up 4: Trusted-Meter Protection Path in MPC Loop
# ==============================================================================
def test_trusted_meter_protection_prevents_adaptive_overload():
    # In run_adaptive_attack_limitation_experiment, passing a trusted meter
    # must actively protect physical transformer headroom
    seed = 42
    feeder_sim = Feeder(transformer_limit_kw=80.0, base_load_profile_kw=[45.0] * 4 + [75.0] * 8 + [45.0] * 4)
    sessions = [
        EVSession(arrival_min=0, departure_min=240, energy_req_kwh=15.0, max_rate_kw=11.0, session_id=f"EV_{i}")
        for i in range(4)
    ]
    sim = ChargingStationSimulator(feeder=feeder_sim, sessions=sessions, time_step_min=15, rng=np.random.default_rng(seed))
    trusted_raw_meter = SimMeter(sim, noise_std_kw=0.1, rng=np.random.default_rng(seed + 999))

    res = run_adaptive_attack_limitation_experiment(seed=seed, trusted_meter=trusted_raw_meter)
    assert res["is_protected"] is True
    assert res["overload_minutes"] == 0.0
    assert res["peak_true_load_kw"] <= res["transformer_limit_kw"]


# ==============================================================================
# Audit Follow-up 5: Validate Every Schedule Before Dispatch (including OPTIMAL)
# ==============================================================================
def test_optimal_schedule_violating_constraints_rejected_and_falls_back():
    s = EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.0, session_id="EV_1")
    feeder = Feeder(transformer_limit_kw=50.0, base_load_profile_kw=[30.0] * 4)
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 4)

    # Mock cp.Problem.solve to return OPTIMAL, but with negative power
    with patch("cvxpy.Problem.solve") as mock_solve, patch("cvxpy.Problem.status", new="optimal"), \
         patch("cvxpy.Variable.value", new=np.array([[-5.0, 5.0, 5.0, 0.0]])):
        mock_solve.return_value = None

        res = solve_charging_schedule(
            sessions=[s],
            feeder=feeder,
            tariff=tariff,
            time_step_min=15,
            num_steps=4,
        )

        # OPTIMAL status with violated constraint must be caught and rejected!
        assert res.is_success is False
        assert "invalid" in res.status
        assert np.all(res.power_schedule_kw == 0.0)


# ==============================================================================
# Audit Follow-up 6: Conditional Protection Claims
# ==============================================================================
def test_conditional_protection_claim():
    # Without independent cap or trusted meter, is_protected must be False
    res_default = run_adaptive_attack_limitation_experiment(seed=42)
    assert res_default["is_protected"] is False
    assert res_default["overload_minutes"] > 0.0

    # With cap, is_protected is True
    res_cap = run_adaptive_attack_limitation_experiment(seed=42, max_aggregate_ev_kw=5.0)
    assert res_cap["is_protected"] is True
    assert res_cap["overload_minutes"] == 0.0


# ==============================================================================
# Audit Follow-up 7: CAN Remote (RTR) and Error Frames Rejected
# ==============================================================================
def test_can_security_remote_and_error_frames_rejected():
    detector = CANSecurityDetector()

    # RTR Remote frame
    msg_remote = can.Message(
        arbitration_id=CAN_ID_BMS_STATUS,
        is_remote_frame=True,
        dlc=8,
        data=b"",
        timestamp=1.0,
    )
    alert_remote = detector.evaluate_message(msg_remote)
    assert alert_remote.is_alarm
    assert alert_remote.alarm_type == "invalid_frame_format"
    assert "Remote" in alert_remote.details
    assert alert_remote.safe_current_setpoint_a == 0.0

    # Error frame
    msg_error = can.Message(
        arbitration_id=CAN_ID_BMS_STATUS,
        is_error_frame=True,
        dlc=8,
        data=b"\x00" * 8,
        timestamp=1.1,
    )
    alert_error = detector.evaluate_message(msg_error)
    assert alert_error.is_alarm
    assert alert_error.alarm_type == "error_frame_detected"
    assert "Error frame" in alert_error.details
    assert alert_error.safe_current_setpoint_a == 0.0


# ==============================================================================
# Audit Hardening 1: Production Mode Rejects Insecure Fallback (CAN FD / 64-bit MAC)
# ==============================================================================
def test_can_security_production_mode_rejects_insecure_fallback(tmp_path: Path):
    # 1. Reject insufficient key length (< 16 bytes)
    with pytest.raises(ValueError, match="at least 16 bytes"):
        CANSecurityDetector(production_mode=True, auth_key=b"too_short_key")

    # 2. Reject missing state_file in production mode
    auth_key_16 = b"0123456789abcdef"  # 16 bytes
    with pytest.raises(ValueError, match="state_file"):
        CANSecurityDetector(production_mode=True, auth_key=auth_key_16, state_file=None)

    state_file = tmp_path / "prod_epoch.txt"
    CANSecurityDetector.initialize_state_file(state_file, initial_epoch=1)
    detector = CANSecurityDetector(production_mode=True, auth_key=auth_key_16, state_file=state_file)

    # 2. Standard 8-byte CAN 2.0 frame must be strictly rejected (insecure protocol fallback)
    msg_can2 = encode_bms_status(
        current_demand_a=80.0,
        voltage_limit_v=380.0,
        soc_percent=50.0,
        pack_temp_c=25.0,
        counter=1,
        auth_key=auth_key_16,
        timestamp=1.0,
    )
    alert_fallback = detector.evaluate_message(msg_can2)
    assert alert_fallback.is_alarm
    assert alert_fallback.alarm_type == "insecure_protocol_fallback"
    assert alert_fallback.safe_current_setpoint_a == 0.0

    # 3. Valid CAN FD frame with 16-byte payload and 64-bit MAC passes
    msg_canfd = encode_bms_status_canfd(
        current_demand_a=85.0,
        voltage_limit_v=380.0,
        soc_percent=52.0,
        pack_temp_c=26.0,
        counter=1,
        auth_key=auth_key_16,
        epoch=detector.boot_epoch,
        timestamp=1.1,
    )
    alert_valid = detector.evaluate_message(msg_canfd)
    assert not alert_valid.is_alarm
    assert alert_valid.safe_current_setpoint_a == 85.0

    # 4. Forged CAN FD payload (wrong MAC with advancing counter=2) is rejected
    tampered_fd = encode_bms_status_canfd(
        current_demand_a=85.0,
        voltage_limit_v=380.0,
        soc_percent=52.0,
        pack_temp_c=26.0,
        counter=2,
        auth_key=b"wrong_key_16byte",
        epoch=detector.boot_epoch,
        timestamp=1.2,
    )
    alert_tampered = detector.evaluate_message(tampered_fd)
    assert alert_tampered.is_alarm
    assert alert_tampered.alarm_type == "auth_failure"
    assert alert_tampered.safe_current_setpoint_a == 0.0


# ==============================================================================
# Audit Hardening 2: Replay Protection Survives Full Process Restart (state_file)
# ==============================================================================
def test_can_security_replay_rejected_across_process_restart_with_state_file(tmp_path: Path):
    auth_key = b"session_secret_key_32bytes_12345"
    state_file = tmp_path / "can_detector_epoch.txt"

    # Detector 1 in process 1: starts at epoch 1
    det1 = CANSecurityDetector(secure_mode=True, auth_key=auth_key, state_file=state_file)
    assert det1.boot_epoch == 1

    msg_proc1 = encode_bms_status(
        current_demand_a=70.0,
        voltage_limit_v=380.0,
        soc_percent=50.0,
        pack_temp_c=25.0,
        counter=1,
        auth_key=auth_key,
        epoch=det1.boot_epoch,
        timestamp=1.0,
    )
    alert1 = det1.evaluate_message(msg_proc1)
    assert not alert1.is_alarm
    assert alert1.safe_current_setpoint_a == 70.0

    # Process 1 terminates; Process 2 starts anew from scratch, passing same state_file
    del det1
    det2 = CANSecurityDetector(secure_mode=True, auth_key=auth_key, state_file=state_file)
    assert det2.boot_epoch == 2  # Monotonically advanced!

    # Attacker replays msg_proc1 from session 1: MAC verification fails under new epoch
    replayed = can.Message(
        arbitration_id=msg_proc1.arbitration_id,
        data=msg_proc1.data,
        is_extended_id=False,
        timestamp=1.1,
    )
    alert_replayed = det2.evaluate_message(replayed)
    assert alert_replayed.is_alarm
    assert alert_replayed.alarm_type == "auth_failure"
    assert alert_replayed.safe_current_setpoint_a == 0.0

    # Fresh frame encoded under new epoch (epoch=2) passes
    msg_proc2 = encode_bms_status(
        current_demand_a=75.0,
        voltage_limit_v=380.0,
        soc_percent=51.0,
        pack_temp_c=25.0,
        counter=2,
        auth_key=auth_key,
        epoch=det2.boot_epoch,
        timestamp=1.2,
    )
    alert_proc2 = det2.evaluate_message(msg_proc2)
    assert not alert_proc2.is_alarm
    assert alert_proc2.safe_current_setpoint_a == 75.0


# ==============================================================================
# Audit Hardening 3: Rejected Frames Cannot Corrupt Timing or Counter State
# ==============================================================================
def test_can_security_rejected_frames_do_not_corrupt_timing_or_counter_state():
    auth_key = b"session_secret_key_32bytes_12345"
    detector = CANSecurityDetector(secure_mode=True, auth_key=auth_key)

    # 1. Process legitimate initial frame at t=1.000, counter=0
    msg0 = encode_bms_status(80.0, 380.0, 50.0, 25.0, timestamp=1.000, counter=0, auth_key=auth_key)
    assert not detector.evaluate_message(msg0).is_alarm
    assert detector.last_msg_timestamp == 1.000
    assert detector.last_counter == 0

    # 2. Attacker injects burst timing anomaly at t=1.015, counter=5
    msg_burst = encode_bms_status(80.0, 380.0, 50.0, 25.0, timestamp=1.015, counter=5, auth_key=auth_key)
    alert_burst = detector.evaluate_message(msg_burst)
    assert alert_burst.is_alarm
    # State MUST NOT be corrupted by rejected frame!
    assert detector.last_msg_timestamp == 1.000
    assert detector.last_counter == 0

    # 3. Attacker injects auth failure frame at t=1.025, counter=10 with invalid tag
    msg_bad_auth = can.Message(
        arbitration_id=CAN_ID_BMS_STATUS,
        data=b"\x00" * 7 + b"\xFF",
        is_extended_id=False,
        timestamp=1.025,
    )
    alert_auth = detector.evaluate_message(msg_bad_auth)
    assert alert_auth.is_alarm
    assert detector.last_msg_timestamp == 1.000
    assert detector.last_counter == 0

    # 4. Attacker injects physical bounds violation (500 A current) at t=1.050, counter=20
    msg_bad_payload = encode_bms_status(500.0, 380.0, 50.0, 25.0, timestamp=1.050, counter=20, auth_key=auth_key)
    alert_payload = detector.evaluate_message(msg_bad_payload)
    assert alert_payload.is_alarm
    assert detector.last_msg_timestamp == 1.000
    assert detector.last_counter == 0

    # 5. Legitimate next frame arrives at nominal time t=1.100, counter=1
    # Delta t = 1.100 - 1.000 = 100 ms (nominal!), delta counter = 1
    msg1 = encode_bms_status(82.0, 380.0, 50.5, 25.2, timestamp=1.100, counter=1, auth_key=auth_key)
    alert1 = detector.evaluate_message(msg1)
    assert not alert1.is_alarm
    assert alert1.safe_current_setpoint_a == 82.0
    assert detector.last_msg_timestamp == 1.100
    assert detector.last_counter == 1


# ==============================================================================
# Audit Hardening 4: Packet Loss, Repeated Frames & Out-of-Order Handling
# ==============================================================================
def test_can_security_freshness_packet_loss_repeated_and_out_of_order():
    auth_key = b"session_secret_key_32bytes_12345"
    detector = CANSecurityDetector(secure_mode=True, auth_key=auth_key)

    # Prime with counter=10 at t=1.000
    msg10 = encode_bms_status(80.0, 380.0, 50.0, 25.0, timestamp=1.000, counter=10, auth_key=auth_key)
    assert not detector.evaluate_message(msg10).is_alarm

    # 1. Repeated frame (counter=10 again) -> Rejected
    msg_repeat = encode_bms_status(80.0, 380.0, 50.0, 25.0, timestamp=1.100, counter=10, auth_key=auth_key)
    alert_repeat = detector.evaluate_message(msg_repeat)
    assert alert_repeat.is_alarm
    assert alert_repeat.alarm_type == "freshness_anomaly"
    assert "Repeated frame" in alert_repeat.details

    # 2. Out-of-order / stale frame (counter=9 < 10) -> Rejected
    msg_out_of_order = encode_bms_status(80.0, 380.0, 50.0, 25.0, timestamp=1.100, counter=9, auth_key=auth_key)
    alert_ooo = detector.evaluate_message(msg_out_of_order)
    assert alert_ooo.is_alarm
    assert alert_ooo.alarm_type == "freshness_anomaly"

    # 3. Excessive jump (> 16): counter jumps from 10 to 30 (delta=20 > 16) -> Rejected
    msg_big_jump = encode_bms_status(80.0, 380.0, 50.0, 25.0, timestamp=1.100, counter=30, auth_key=auth_key)
    alert_big_jump = detector.evaluate_message(msg_big_jump)
    assert alert_big_jump.is_alarm
    assert alert_big_jump.alarm_type == "freshness_anomaly"

    # 4. Legitimate packet loss (delta=3 <= 16): counter jumps from 10 to 13 at t=1.300 (3 intervals) -> Accepted
    msg_loss = encode_bms_status(80.0, 380.0, 50.0, 25.0, timestamp=1.300, counter=13, auth_key=auth_key)
    alert_loss = detector.evaluate_message(msg_loss)
    assert not alert_loss.is_alarm
    assert alert_loss.safe_current_setpoint_a == 80.0


# ==============================================================================
# Audit Hardening 5: Trusted-Meter Validation, Noise & Actuation
# ==============================================================================
class FaultyTrustedMeter:
    """Mock meter that returns invalid, negative, or erroneous readings."""

    def __init__(self, fault_type: str):
        self.fault_type = fault_type

    def read_feeder_kw(self, t_min: int) -> float:
        if self.fault_type == "nan":
            return float("nan")
        elif self.fault_type == "negative":
            return -15.0
        elif self.fault_type == "none":
            return None  # type: ignore
        elif self.fault_type == "exception":
            raise RuntimeError("Hardware communication timeout")
        return 50.0


def test_trusted_meter_input_validation_and_noise():
    feeder = Feeder(transformer_limit_kw=80.0, base_load_profile_kw=[40.0] * 4)
    sessions = [EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.0, session_id="EV_1")]
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 4)
    sim = ChargingStationSimulator(feeder=feeder, sessions=sessions, time_step_min=15)
    evse = SimEVSE(sim)
    primary_meter = SimMeter(sim)

    # 1. Faulty meter tests: NaN, negative, None, exception must be caught and trigger safe mode without crashing
    for fault in ["nan", "negative", "none", "exception"]:
        faulty = FaultyTrustedMeter(fault)
        res = run_mpc_charging_loop(
            feeder=feeder,
            sessions=sessions,
            tariff=tariff,
            meter=primary_meter,
            evse=evse,
            simulator=sim,
            time_step_min=15,
            num_steps=4,
            trusted_meter=faulty,  # type: ignore
        )
        assert any(res.alarms_raised)  # Flagged alarm defensively
        assert any(res.safe_mode_active)  # Entered safe mode

    # 2. Noisy trusted meter (std = 1.0 kW): loop completes and maintains safe physical operation
    noisy_trusted = SimMeter(sim, noise_std_kw=1.0, rng=np.random.default_rng(999))
    res_noisy = run_mpc_charging_loop(
        feeder=feeder,
        sessions=sessions,
        tariff=tariff,
        meter=primary_meter,
        evse=evse,
        simulator=sim,
        time_step_min=15,
        num_steps=4,
        trusted_meter=noisy_trusted,
    )
    assert max(res_noisy.true_feeder_kw) <= feeder.transformer_limit_kw + 0.05


# ==============================================================================
# Audit Hardening 6: Dynamic Base-Load Surge & Already-Overloaded Feeder
# ==============================================================================
def test_trusted_meter_dynamic_base_load_surge_and_already_overloaded_feeder():
    # 1. Dynamic surge: base load jumps from 30 kW to 72 kW at step 2 (t=30 min)
    actual_base = [30.0, 30.0, 72.0, 72.0]
    limit = 80.0
    feeder_sim = Feeder(transformer_limit_kw=limit, base_load_profile_kw=actual_base)
    feeder_ctrl = Feeder(transformer_limit_kw=limit, base_load_profile_kw=[30.0] * 4)
    sessions = [EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=15.0, session_id="EV_SURGE")]
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 4)

    sim = ChargingStationSimulator(feeder=feeder_sim, sessions=sessions, time_step_min=15)
    evse = SimEVSE(sim)
    trusted = SimMeter(sim)

    res_surge = run_mpc_charging_loop(
        feeder=feeder_ctrl,
        sessions=sessions,
        tariff=tariff,
        meter=SimMeter(sim),
        evse=evse,
        simulator=sim,
        time_step_min=15,
        num_steps=4,
        trusted_meter=trusted,
        conservative_margin_kw=5.0,
    )
    # Physical load must never exceed transformer rating
    assert max(res_surge.true_feeder_kw) <= limit + 0.05

    # 2. Already overloaded feeder (P_base = 85 kW > 80 kW limit)
    base_over = [85.0] * 4
    feeder_over_sim = Feeder(transformer_limit_kw=limit, base_load_profile_kw=base_over)
    feeder_over_ctrl = Feeder(transformer_limit_kw=limit, base_load_profile_kw=base_over)
    sessions_over = [EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=15.0, session_id="EV_OVER")]

    sim_over = ChargingStationSimulator(feeder=feeder_over_sim, sessions=sessions_over, time_step_min=15)
    evse_over = SimEVSE(sim_over)
    trusted_over = SimMeter(sim_over)

    res_over = run_mpc_charging_loop(
        feeder=feeder_over_ctrl,
        sessions=sessions_over,
        tariff=tariff,
        meter=SimMeter(sim_over),
        evse=evse_over,
        simulator=sim_over,
        time_step_min=15,
        num_steps=4,
        trusted_meter=trusted_over,
    )
    # Controller must throttle all EVs to 0.0 kW when feeder is already overloaded
    assert all(p == 0.0 for p in res_over.total_ev_power_kw)
    assert evse_over.read_session_kw("EV_OVER") == 0.0


# ==============================================================================
# Audit Hardening 7: validate_schedule_constraints() Hardened Validations
# ==============================================================================
def test_validate_schedule_constraints_hardened_inputs():
    s = EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.0, session_id="EV_1")
    power_matrix = np.zeros((1, 4))
    unmet_arr = np.array([10.0])
    base_load = np.array([20.0, 20.0, 20.0, 20.0])

    # 1. Invalid dt_hours
    ok, msg = validate_schedule_constraints(power_matrix, unmet_arr, [s], base_load, 50.0, dt_hours=0.0)
    assert not ok and "dt_hours" in msg
    ok, msg = validate_schedule_constraints(power_matrix, unmet_arr, [s], base_load, 50.0, dt_hours=-0.25)
    assert not ok and "dt_hours" in msg
    ok, msg = validate_schedule_constraints(power_matrix, unmet_arr, [s], base_load, 50.0, dt_hours=float("nan"))
    assert not ok and "dt_hours" in msg

    # 2. Invalid transformer limit
    ok, msg = validate_schedule_constraints(power_matrix, unmet_arr, [s], base_load, transformer_limit_kw=0.0, dt_hours=0.25)
    assert not ok and "transformer_limit_kw" in msg
    ok, msg = validate_schedule_constraints(power_matrix, unmet_arr, [s], base_load, transformer_limit_kw=-50.0, dt_hours=0.25)
    assert not ok and "transformer_limit_kw" in msg

    # 3. Invalid tolerances
    ok, msg = validate_schedule_constraints(power_matrix, unmet_arr, [s], base_load, 50.0, 0.25, tol_kw=-0.1)
    assert not ok and "Tolerance parameters" in msg
    ok, msg = validate_schedule_constraints(power_matrix, unmet_arr, [s], base_load, 50.0, 0.25, tol_kwh=float("nan"))
    assert not ok and "Tolerance parameters" in msg

    # 4. Malformed base load profile
    ok, msg = validate_schedule_constraints(power_matrix, unmet_arr, [s], [20.0, 20.0], 50.0, 0.25)  # type: ignore
    assert not ok and "numpy ndarray" in msg
    ok, msg = validate_schedule_constraints(power_matrix, unmet_arr, [s], np.zeros((2, 2)), 50.0, 0.25)
    assert not ok and "1D array" in msg
    ok, msg = validate_schedule_constraints(power_matrix, unmet_arr, [s], np.array([20.0, float("nan"), 20.0, 20.0]), 50.0, 0.25)
    assert not ok and "Non-finite" in msg
    ok, msg = validate_schedule_constraints(power_matrix, unmet_arr, [s], np.array([20.0, 20.0]), 50.0, 0.25, num_steps=4)  # Wrong shape
    assert not ok and "base load shape" in msg


# ==============================================================================
# Audit Hardening 8: Production CAN Mode Fails Closed on Missing/Corrupt State File
# ==============================================================================
def test_production_can_mode_fails_closed_on_missing_or_corrupt_state_file(tmp_path: Path):
    auth_key_16 = b"0123456789abcdef"
    missing_file = tmp_path / "non_existent_epoch.txt"

    # Missing state file in production mode must fail closed (FileNotFoundError)
    with pytest.raises(FileNotFoundError, match="Production state file missing"):
        CANSecurityDetector(production_mode=True, auth_key=auth_key_16, state_file=missing_file)

    # Corrupt state file (non-integer) must fail closed (RuntimeError)
    corrupt_file = tmp_path / "corrupt_epoch.txt"
    corrupt_file.write_text("CORRUPTED_NOT_INT\n")
    with pytest.raises(RuntimeError, match="Corrupt or unreadable epoch state file"):
        CANSecurityDetector(production_mode=True, auth_key=auth_key_16, state_file=corrupt_file)

    # Corrupt state file (negative integer) must fail closed (RuntimeError)
    corrupt_neg_file = tmp_path / "corrupt_neg_epoch.txt"
    corrupt_neg_file.write_text("-5\n")
    with pytest.raises(RuntimeError, match="Corrupt or unreadable epoch state file"):
        CANSecurityDetector(production_mode=True, auth_key=auth_key_16, state_file=corrupt_neg_file)

    # Valid state file initializes and increments across restarts
    valid_file = tmp_path / "valid_epoch.txt"
    CANSecurityDetector.initialize_state_file(valid_file, initial_epoch=1)

    # Process 1: detector loads epoch 1, increments and saves epoch 2
    det1 = CANSecurityDetector(production_mode=True, auth_key=auth_key_16, state_file=valid_file)
    assert det1.boot_epoch == 2

    # Frame generated with epoch 1 is rejected as stale/forged
    msg_epoch1 = encode_bms_status_canfd(
        current_demand_a=50.0,
        voltage_limit_v=380.0,
        soc_percent=50.0,
        pack_temp_c=25.0,
        counter=1,
        auth_key=auth_key_16,
        epoch=1,
        timestamp=1.0,
    )
    alert1 = det1.evaluate_message(msg_epoch1)
    assert alert1.is_alarm
    assert alert1.alarm_type == "auth_failure"

    # Frame generated with epoch 2 is accepted
    msg_epoch2 = encode_bms_status_canfd(
        current_demand_a=50.0,
        voltage_limit_v=380.0,
        soc_percent=50.0,
        pack_temp_c=25.0,
        counter=1,
        auth_key=auth_key_16,
        epoch=2,
        timestamp=1.0,
    )
    alert2 = det1.evaluate_message(msg_epoch2)
    assert not alert2.is_alarm

    # Process 2 (simulated restart): detector loads epoch 2, increments and saves epoch 3
    det2 = CANSecurityDetector(production_mode=True, auth_key=auth_key_16, state_file=valid_file)
    assert det2.boot_epoch == 3

    # Replaying frame from epoch 2 against det2 must be rejected
    alert3 = det2.evaluate_message(msg_epoch2)
    assert alert3.is_alarm
    assert alert3.alarm_type == "auth_failure"


# ==============================================================================
# Audit Hardening 9: CAN Timestamps Validation and Backward Handling
# ==============================================================================
def test_can_timestamps_validation_and_backward_handling():
    detector = CANSecurityDetector(expected_period_s=0.100, timing_tolerance_s=0.040)

    # Baseline valid frame at t = 1.0 s
    msg_baseline = encode_bms_status(
        current_demand_a=40.0,
        voltage_limit_v=380.0,
        soc_percent=50.0,
        pack_temp_c=25.0,
        counter=1,
        timestamp=1.0,
    )
    alert0 = detector.evaluate_message(msg_baseline)
    assert not alert0.is_alarm
    assert detector.last_msg_timestamp == 1.0
    assert detector.last_counter == 1

    # 1. NaN timestamp
    msg_nan = encode_bms_status(
        current_demand_a=40.0,
        voltage_limit_v=380.0,
        soc_percent=50.0,
        pack_temp_c=25.0,
        counter=2,
        timestamp=float("nan"),
    )
    alert_nan = detector.evaluate_message(msg_nan)
    assert alert_nan.is_alarm
    assert alert_nan.alarm_type == "invalid_timestamp"
    # State must be unchanged
    assert detector.last_msg_timestamp == 1.0
    assert detector.last_counter == 1

    # 2. Positive infinity timestamp
    msg_inf = encode_bms_status(
        current_demand_a=40.0,
        voltage_limit_v=380.0,
        soc_percent=50.0,
        pack_temp_c=25.0,
        counter=2,
        timestamp=float("inf"),
    )
    alert_inf = detector.evaluate_message(msg_inf)
    assert alert_inf.is_alarm
    assert alert_inf.alarm_type == "invalid_timestamp"
    assert detector.last_msg_timestamp == 1.0
    assert detector.last_counter == 1

    # 3. Negative infinity timestamp
    msg_neginf = encode_bms_status(
        current_demand_a=40.0,
        voltage_limit_v=380.0,
        soc_percent=50.0,
        pack_temp_c=25.0,
        counter=2,
        timestamp=float("-inf"),
    )
    alert_neginf = detector.evaluate_message(msg_neginf)
    assert alert_neginf.is_alarm
    assert alert_neginf.alarm_type == "invalid_timestamp"
    assert detector.last_msg_timestamp == 1.0
    assert detector.last_counter == 1

    # 4. Negative timestamp
    msg_neg = encode_bms_status(
        current_demand_a=40.0,
        voltage_limit_v=380.0,
        soc_percent=50.0,
        pack_temp_c=25.0,
        counter=2,
        timestamp=-0.5,
    )
    alert_neg = detector.evaluate_message(msg_neg)
    assert alert_neg.is_alarm
    assert alert_neg.alarm_type == "invalid_timestamp"
    assert detector.last_msg_timestamp == 1.0
    assert detector.last_counter == 1

    # 5. Backward timestamp (current 0.9 s < last 1.0 s)
    msg_backward = encode_bms_status(
        current_demand_a=40.0,
        voltage_limit_v=380.0,
        soc_percent=50.0,
        pack_temp_c=25.0,
        counter=2,
        timestamp=0.9,
    )
    alert_back = detector.evaluate_message(msg_backward)
    assert alert_back.is_alarm
    assert alert_back.alarm_type == "timing_anomaly"
    assert "Backward timestamp" in alert_back.details
    assert detector.last_msg_timestamp == 1.0
    assert detector.last_counter == 1

    # 6. Subsequent valid frame at t = 1.1 s passes cleanly
    msg_valid = encode_bms_status(
        current_demand_a=40.0,
        voltage_limit_v=380.0,
        soc_percent=50.0,
        pack_temp_c=25.0,
        counter=2,
        timestamp=1.1,
    )
    alert_valid = detector.evaluate_message(msg_valid)
    assert not alert_valid.is_alarm
    assert detector.last_msg_timestamp == 1.1
    assert detector.last_counter == 2


# ==============================================================================
# Audit Hardening 10: Unrelated CAN Traffic Does Not Trigger EVSE Fail-Safe
# ==============================================================================
def test_unrelated_can_traffic_does_not_trigger_evse_failsafe():
    feeder = Feeder(transformer_limit_kw=50.0, base_load_profile_kw=[10.0] * 4)
    sessions = [EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.0, session_id="EV_CAN")]
    sim = ChargingStationSimulator(feeder=feeder, sessions=sessions, time_step_min=15)
    evse = SimEVSE(sim)
    evse.set_setpoint_kw("EV_CAN", 6.6)
    assert evse.read_session_kw("EV_CAN") == 6.6

    detector = CANSecurityDetector(evse=evse, ev_id="EV_CAN")

    # 1. Unrelated extended 29-bit CAN ID 0x200 (e.g. Inverter broadcast)
    msg_unrelated_ext = can.Message(
        arbitration_id=0x200,
        is_extended_id=True,
        data=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        timestamp=1.0,
    )
    alert1 = detector.evaluate_message(msg_unrelated_ext)
    assert not alert1.is_alarm
    # EVSE setpoint must remain active at 6.6 kW!
    assert evse.read_session_kw("EV_CAN") == 6.6

    # 2. Unrelated RTR remote frame with arbitration ID 0x300
    msg_unrelated_rtr = can.Message(
        arbitration_id=0x300,
        is_remote_frame=True,
        dlc=0,
        timestamp=1.1,
    )
    alert2 = detector.evaluate_message(msg_unrelated_rtr)
    assert not alert2.is_alarm
    assert evse.read_session_kw("EV_CAN") == 6.6

    # 3. BMS status frame with extended ID 29-bit MUST trigger alarm and actuate fail-safe to 0.0 kW
    msg_bms_ext = can.Message(
        arbitration_id=CAN_ID_BMS_STATUS,
        is_extended_id=True,
        data=b"\x00" * 8,
        timestamp=1.2,
    )
    alert3 = detector.evaluate_message(msg_bms_ext)
    assert alert3.is_alarm
    assert alert3.alarm_type == "invalid_frame_format"
    assert evse.read_session_kw("EV_CAN") == 0.0

    # Reset EVSE setpoint to 6.6 kW for next check
    evse.set_setpoint_kw("EV_CAN", 6.6)
    assert evse.read_session_kw("EV_CAN") == 6.6

    # 4. BMS status frame with RTR MUST trigger alarm and actuate fail-safe to 0.0 kW
    msg_bms_rtr = can.Message(
        arbitration_id=CAN_ID_BMS_STATUS,
        is_remote_frame=True,
        dlc=0,
        timestamp=1.3,
    )
    alert4 = detector.evaluate_message(msg_bms_rtr)
    assert alert4.is_alarm
    assert alert4.alarm_type == "invalid_frame_format"
    assert evse.read_session_kw("EV_CAN") == 0.0


# ==============================================================================
# Audit Hardening 11: Trusted-Meter Failure Paths & Deactivated Protection
# ==============================================================================
class StaleTrustedMeter:
    def __init__(self, is_stale: bool = True):
        self.is_stale = is_stale

    def read_feeder_kw(self, t_min: int) -> float:
        return 40.0


def test_trusted_meter_failure_paths_use_zero_charging_and_deactivate_protection():
    feeder = Feeder(transformer_limit_kw=80.0, base_load_profile_kw=[40.0] * 4)
    sessions = [EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.0, session_id="EV_FAIL")]
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 4)
    sim = ChargingStationSimulator(feeder=feeder, sessions=sessions, time_step_min=15)
    evse = SimEVSE(sim)
    primary_meter = SimMeter(sim)

    # 1. Test failure paths with NO aggregate cap (must fall back to zero charging 0.0 kW)
    failure_meters = [
        FaultyTrustedMeter("nan"),
        FaultyTrustedMeter("negative"),
        FaultyTrustedMeter("none"),
        FaultyTrustedMeter("exception"),
        StaleTrustedMeter(is_stale=True),
    ]

    for faulty_meter in failure_meters:
        res = run_mpc_charging_loop(
            feeder=feeder,
            sessions=sessions,
            tariff=tariff,
            meter=primary_meter,
            evse=evse,
            simulator=sim,
            time_step_min=15,
            num_steps=4,
            trusted_meter=faulty_meter,  # type: ignore
            max_aggregate_ev_kw=None,
        )
        # Protection must NOT be reported as active because defense failed
        assert res.is_protection_active is False
        # All EV power must be throttled to 0.0 kW (zero charging fallback)
        assert all(p == 0.0 for p in res.total_ev_power_kw)

        # In limitation study, run_adaptive_attack_limitation_experiment must report is_protected=False
        exp_res = run_adaptive_attack_limitation_experiment(trusted_meter=faulty_meter)  # type: ignore
        assert exp_res["is_protected"] is False

    # 2. When an independent aggregate cap is provided, it falls back to that cap and protects
    res_cap = run_mpc_charging_loop(
        feeder=feeder,
        sessions=sessions,
        tariff=tariff,
        meter=primary_meter,
        evse=evse,
        simulator=sim,
        time_step_min=15,
        num_steps=4,
        trusted_meter=FaultyTrustedMeter("nan"),  # type: ignore
        max_aggregate_ev_kw=5.0,
    )
    assert res_cap.is_protection_active is True
    assert max(res_cap.total_ev_power_kw) <= 5.0 + 1e-4


# ==============================================================================
# Audit Hardening 12: Raw Solver Outputs Cannot Be Hidden by Clipping
# ==============================================================================
def test_raw_solver_outputs_cannot_be_hidden_by_clipping():
    sessions = [EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.0, session_id="EV_CLIP")]
    feeder = Feeder(transformer_limit_kw=50.0, base_load_profile_kw=[20.0] * 4)
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 4)

    def fake_solve(self, *args, **kwargs):
        for var in self.variables():
            if var.name() == "charging_power_kw":
                var.save_value(np.array([[-2.0, 5.0, 5.0, 5.0]]))
            elif var.name() == "unmet_energy_kwh":
                var.save_value(np.array([0.0]))
        self._status = "optimal"
        return 10.0

    with patch.object(cp.Problem, "solve", fake_solve):
        res = solve_charging_schedule(
            sessions=sessions,
            feeder=feeder,
            tariff=tariff,
            time_step_min=15,
            num_steps=4,
        )
        # Must reject raw solution before any clipping can happen!
        assert res.is_success is False
        assert "invalid" in res.status
        assert np.all(res.power_schedule_kw == 0.0)


# ==============================================================================
# Audit Hardening 13: Rollback Rejection on reset() and state_file Monotonicity
# ==============================================================================
def test_epoch_rollback_rejected(tmp_path: Path):
    auth_key_16 = b"0123456789abcdef"
    state_file = tmp_path / "rollback_epoch.txt"
    CANSecurityDetector.initialize_state_file(state_file, initial_epoch=5)

    det = CANSecurityDetector(production_mode=True, auth_key=auth_key_16, state_file=state_file)
    # Starts at incremented epoch = 6
    assert det.boot_epoch == 6
    assert det.get_active_epoch() == 6

    # 1. Reset with smaller epoch must be rejected (ValueError)
    with pytest.raises(ValueError, match="Epoch rollback rejected"):
        det.reset(new_epoch=3)

    # 2. Reset with equal epoch must be rejected (ValueError)
    with pytest.raises(ValueError, match="Epoch rollback rejected"):
        det.reset(new_epoch=6)

    # 3. Reset with strictly greater epoch succeeds
    det.reset(new_epoch=10)
    assert det.boot_epoch == 10
    assert det.get_active_epoch() == 10

    # 4. Attempting to directly write an older epoch into state_file raises ValueError
    with pytest.raises(RuntimeError, match="Rollback rejected"):
        det._save_epoch(4)


# ==============================================================================
# Audit Hardening 14: Concurrent Detector Startup Serialization
# ==============================================================================
def test_concurrent_detector_initialization(tmp_path: Path):
    from concurrent.futures import ThreadPoolExecutor

    auth_key_16 = b"0123456789abcdef"
    state_file = tmp_path / "concurrent_epoch.txt"
    CANSecurityDetector.initialize_state_file(state_file, initial_epoch=1)

    num_workers = 10

    def init_worker(idx: int) -> int:
        detector = CANSecurityDetector(
            production_mode=True,
            auth_key=auth_key_16,
            state_file=state_file,
        )
        return detector.boot_epoch

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        epochs = list(executor.map(init_worker, range(num_workers)))

    # All initializations must have been serialized and selected strictly unique, monotonic epochs
    assert len(epochs) == num_workers
    assert len(set(epochs)) == num_workers
    assert min(epochs) == 2
    assert max(epochs) == 1 + num_workers

    # Persisted epoch in state file must equal highest assigned epoch
    final_stored = int(state_file.read_text(encoding="utf-8").strip())
    assert final_stored == 1 + num_workers


# ==============================================================================
# Audit Hardening 15: Trusted-Meter Freshness Contract & Stale Fallback
# ==============================================================================
class UnverifiedLegacyMeter(MeterInterface):
    """Meter that does not implement sample timestamps or freshness checks."""

    def read_feeder_kw(self, t_min: int) -> float:
        return 40.0


class ExpiredTimestampMeter(MeterInterface):
    """Meter with expired sample timestamps (older than 15-minute threshold)."""

    def read_feeder_kw(self, t_min: int) -> float:
        return 40.0

    def get_reading_timestamp_s(self, t_min: int) -> float | None:
        # Timestamp is 30 minutes in the past (age = 1800 s > 900 s)
        return float((t_min - 30) * 60.0)


def test_trusted_meter_freshness_contract_and_stale_fallback():
    feeder = Feeder(transformer_limit_kw=80.0, base_load_profile_kw=[40.0] * 4)
    sessions = [EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.0, session_id="EV_STALE")]
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 4)
    sim = ChargingStationSimulator(feeder=feeder, sessions=sessions, time_step_min=15)
    evse = SimEVSE(sim)
    primary_meter = SimMeter(sim)

    # 1. Meter without timestamps or freshness verification cannot establish freshness
    unverified_meter = UnverifiedLegacyMeter()
    assert unverified_meter.is_reading_fresh(0) is False

    res_unverified = run_mpc_charging_loop(
        feeder=feeder,
        sessions=sessions,
        tariff=tariff,
        meter=primary_meter,
        evse=evse,
        simulator=sim,
        time_step_min=15,
        num_steps=4,
        trusted_meter=unverified_meter,
        max_aggregate_ev_kw=None,
    )
    # Must fall back to zero charging and report protection inactive
    assert res_unverified.is_protection_active is False
    assert all(p == 0.0 for p in res_unverified.total_ev_power_kw)
    assert any(res_unverified.alarms_raised)

    # 2. Meter with expired timestamp (older than 15 minutes) is rejected as stale
    expired_meter = ExpiredTimestampMeter()
    assert expired_meter.is_reading_fresh(30, max_age_s=900.0) is False

    res_expired = run_mpc_charging_loop(
        feeder=feeder,
        sessions=sessions,
        tariff=tariff,
        meter=primary_meter,
        evse=evse,
        simulator=sim,
        time_step_min=15,
        num_steps=4,
        trusted_meter=expired_meter,
        max_aggregate_ev_kw=None,
    )
    assert res_expired.is_protection_active is False
    assert all(p == 0.0 for p in res_expired.total_ev_power_kw)


# ==============================================================================
# Audit Hardening 16: EVSE Command Exceptions & Explicit Actuation Status
# ==============================================================================
class FaultySetPointEVSE(EVSEInterface):
    """EVSE whose setpoint command fails with a hardware timeout."""

    def __init__(self, sim: Any) -> None:
        self._sim = sim

    def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
        raise RuntimeError("I2C bus timeout to charger power controller")

    def read_session_kw(self, ev_id: str) -> float:
        return 7.0


class StuckRelayEVSE(EVSEInterface):
    """EVSE that accepts setpoint command but power relay remains stuck at 6.0 kW."""

    def __init__(self, sim: Any) -> None:
        self._sim = sim

    def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
        pass  # Acknowledged command

    def read_session_kw(self, ev_id: str) -> float:
        return 6.0  # Welded contactor delivering 6.0 kW


def test_evse_command_exceptions_and_actuation_status():
    feeder = Feeder(transformer_limit_kw=50.0, base_load_profile_kw=[10.0] * 4)
    sessions = [EVSession(arrival_min=0, departure_min=60, energy_req_kwh=10.0, max_rate_kw=7.0, session_id="EV_ACT")]
    sim = ChargingStationSimulator(feeder=feeder, sessions=sessions, time_step_min=15)

    malformed_msg = can.Message(
        arbitration_id=CAN_ID_BMS_STATUS,
        data=b"\x00" * 4,  # Malformed payload (4 bytes instead of 8)
        is_extended_id=False,
        timestamp=1.0,
    )

    # 1. Faulty setpoint dispatch: catches exception and reports status='failed'
    faulty_evse = FaultySetPointEVSE(sim)
    det_fault = CANSecurityDetector(evse=faulty_evse, ev_id="EV_ACT")
    alert_fault = det_fault.evaluate_message(malformed_msg)
    assert alert_fault.is_alarm is True
    assert alert_fault.actuation_status == "failed"
    assert alert_fault.readback_power_kw is None

    # 2. Stuck relay / unverified setpoint: readback reports 6.0 kW != 0.0 kW
    stuck_evse = StuckRelayEVSE(sim)
    det_stuck = CANSecurityDetector(evse=stuck_evse, ev_id="EV_ACT")
    alert_stuck = det_stuck.evaluate_message(malformed_msg)
    assert alert_stuck.is_alarm is True
    assert alert_stuck.actuation_status == "unverified"
    assert alert_stuck.readback_power_kw == 6.0

    # 3. Normal EVSE: verified confirmed zero power
    normal_evse = SimEVSE(sim)
    det_normal = CANSecurityDetector(evse=normal_evse, ev_id="EV_ACT")
    alert_normal = det_normal.evaluate_message(malformed_msg)
    assert alert_normal.is_alarm is True
    assert alert_normal.actuation_status == "confirmed_zero"
    assert alert_normal.readback_power_kw == 0.0

    # 4. Advisory mode (no EVSE connected): reports 'not_connected'
    det_advisory = CANSecurityDetector(evse=None, ev_id=None)
    alert_advisory = det_advisory.evaluate_message(malformed_msg)
    assert alert_advisory.is_alarm is True
    assert alert_advisory.actuation_status == "not_connected"
    assert alert_advisory.readback_power_kw is None


# ==============================================================================
# Audit Hardening 17: Supported CAN FD Top-Level Package Export
# ==============================================================================
def test_can_fd_top_level_import_path():
    # Verify top-level exports from can_security
    import can_security

    assert hasattr(can_security, "encode_bms_status_canfd")
    assert hasattr(can_security, "decode_bms_status_canfd")
    assert hasattr(can_security, "compute_auth_tag_64bit")

    from can_security import decode_bms_status_canfd, encode_bms_status_canfd

    msg = encode_bms_status_canfd(
        current_demand_a=60.0,
        voltage_limit_v=390.0,
        soc_percent=65.0,
        pack_temp_c=28.0,
        counter=42,
        auth_key=b"session_key_16bytes!",
        epoch=3,
        timestamp=2.5,
    )
    assert msg.is_fd is True
    assert len(msg.data) == 16

    decoded = decode_bms_status_canfd(msg)
    assert decoded is not None
    assert pytest.approx(decoded.current_demand_a, 0.1) == 60.0
    assert pytest.approx(decoded.voltage_limit_v, 0.1) == 390.0
    assert pytest.approx(decoded.soc_percent, 0.5) == 65.0
    assert pytest.approx(decoded.pack_temp_c, 1.0) == 28.0
    assert decoded.counter == 42


# ==============================================================================
# Audit Hardening 18: Safe State File Initialization & First-Time Race Protection
# ==============================================================================
def test_initialize_state_file_preserves_existing_epoch_and_fails_if_exists(tmp_path: Path):
    """Test initialize_state_file() fails closed on existing files without truncating/resetting."""
    state_file = tmp_path / "preserved_epoch.txt"
    CANSecurityDetector.initialize_state_file(state_file, initial_epoch=5)
    assert int(state_file.read_text(encoding="utf-8").strip()) == 5

    # Attempting to re-initialize an existing state file must raise FileExistsError
    with pytest.raises(FileExistsError, match="already exists"):
        CANSecurityDetector.initialize_state_file(state_file, initial_epoch=1)

    # State file contents must remain strictly preserved (never truncated or modified)
    assert int(state_file.read_text(encoding="utf-8").strip()) == 5

    with pytest.raises(FileExistsError, match="already exists"):
        CANSecurityDetector.initialize_state_file(state_file, initial_epoch=10)

    assert int(state_file.read_text(encoding="utf-8").strip()) == 5


def test_initialize_state_file_validates_initial_epoch(tmp_path: Path):
    """Test initialize_state_file() rejects non-positive initial epochs."""
    state_file = tmp_path / "invalid_init.txt"
    with pytest.raises(ValueError, match="Initial epoch must be >= 1"):
        CANSecurityDetector.initialize_state_file(state_file, initial_epoch=0)

    with pytest.raises(ValueError, match="Initial epoch must be >= 1"):
        CANSecurityDetector.initialize_state_file(state_file, initial_epoch=-5)

    assert not state_file.exists()


def test_concurrent_first_time_detector_initialization(tmp_path: Path):
    """Test that concurrent first-time detector startups serialize and cannot assign duplicate epochs."""
    from concurrent.futures import ThreadPoolExecutor

    auth_key_16 = b"0123456789abcdef"
    state_file = tmp_path / "fresh_race_epoch.txt"
    assert not state_file.exists()

    num_workers = 10

    def start_worker(idx: int) -> int:
        detector = CANSecurityDetector(
            secure_mode=True,
            auth_key=auth_key_16,
            state_file=state_file,
        )
        return detector.boot_epoch

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        epochs = list(executor.map(start_worker, range(num_workers)))

    # First-time concurrent initialization must have been serialized:
    # Exactly one detector got 1, and the others got unique, monotonic increments 2..10
    assert len(epochs) == num_workers
    assert len(set(epochs)) == num_workers, f"Duplicate epochs assigned: {epochs}"
    assert sorted(epochs) == list(range(1, 1 + num_workers))

    # Persisted state file must hold the final incremented epoch
    final_stored = int(state_file.read_text(encoding="utf-8").strip())
    assert final_stored == num_workers


def test_concurrent_first_time_initialize_state_file(tmp_path: Path):
    """Test that concurrent initialize_state_file() calls serialize and exactly one succeeds."""
    from concurrent.futures import ThreadPoolExecutor

    state_file = tmp_path / "concurrent_provision.txt"
    assert not state_file.exists()

    num_workers = 10
    results: list[str] = []

    def provision_worker(idx: int) -> str:
        try:
            CANSecurityDetector.initialize_state_file(state_file, initial_epoch=42)
            return "success"
        except FileExistsError:
            return "exists_error"

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(provision_worker, range(num_workers)))

    # Exactly one process/thread succeeded in creation; the other 9 caught FileExistsError
    assert results.count("success") == 1
    assert results.count("exists_error") == num_workers - 1

    # Stored content is uncorrupted and contains initial_epoch
    assert int(state_file.read_text(encoding="utf-8").strip()) == 42





