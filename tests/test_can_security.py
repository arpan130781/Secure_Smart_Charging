"""Unit tests for CAN bus security module (Task 19).

Tests:
1. Normal periodic BMS status decoding and timing check.
2. Value-tampering attack detection via payload range check.
3. Replay attack detection via timing anomaly check.
4. Python-can virtual bus integration.
"""

import can
import pytest

from can_security.bms_protocol import (
    CAN_ID_BMS_STATUS,
    decode_bms_status,
    encode_bms_status,
)
from can_security.can_attacks import (
    inject_replay_attack,
    inject_value_tampering,
)
from can_security.can_detector import CANSecurityDetector
from ev_sched.hal import EVSEInterface


def test_encode_and_decode_bms_status() -> None:
    msg = encode_bms_status(
        current_demand_a=125.4,
        voltage_limit_v=398.2,
        soc_percent=68.5,
        pack_temp_c=28.0,
        timestamp=10.500,
    )

    assert msg.arbitration_id == CAN_ID_BMS_STATUS
    assert len(msg.data) == 8

    decoded = decode_bms_status(msg)
    assert decoded is not None
    assert pytest.approx(decoded.current_demand_a, 0.1) == 125.4
    assert pytest.approx(decoded.voltage_limit_v, 0.1) == 398.2
    assert pytest.approx(decoded.soc_percent, 0.5) == 68.5
    assert pytest.approx(decoded.pack_temp_c, 1.0) == 28.0
    assert decoded.timestamp == 10.500

    # Test extreme negative temperature clamping: -45 C clamps to -40 C dynamic floor without wrap-around
    msg_cold = encode_bms_status(50.0, 380.0, 50.0, -45.0, timestamp=1.0)
    decoded_cold = decode_bms_status(msg_cold)
    assert decoded_cold is not None
    assert decoded_cold.pack_temp_c == -40.0


def test_can_detector_normal_periodic_traffic() -> None:
    detector = CANSecurityDetector(expected_period_s=0.100, timing_tolerance_s=0.030)

    # Sequence of 20 normal messages at 100 ms intervals (0.100 s, 0.200 s, ...)
    for i in range(20):
        t = 1.0 + i * 0.100
        msg = encode_bms_status(
            current_demand_a=80.0,
            voltage_limit_v=380.0,
            soc_percent=50.0,
            pack_temp_c=25.0,
            timestamp=t,
        )
        alert = detector.evaluate_message(msg)
        assert not alert.is_alarm
        assert alert.alarm_type is None


def test_can_detector_detects_value_tampering() -> None:
    detector = CANSecurityDetector(max_allowable_current_a=200.0, max_voltage_v=450.0)

    # Legitimate frame
    legit_msg = encode_bms_status(
        current_demand_a=90.0,
        voltage_limit_v=380.0,
        soc_percent=50.0,
        pack_temp_c=25.0,
        timestamp=2.000,
    )
    alert_legit = detector.evaluate_message(legit_msg)
    assert not alert_legit.is_alarm

    # Attacked frame: value tampered to 550.0 A (dangerously exceeding 200 A ceiling)
    tampered_msg = inject_value_tampering(
        legit_msg, tampered_current_a=550.0, tampered_voltage_v=380.0
    )
    tampered_msg.timestamp = 2.100  # Normal timing, but malicious payload

    alert_tampered = detector.evaluate_message(tampered_msg)
    assert alert_tampered.is_alarm
    assert alert_tampered.alarm_type == "payload_out_of_bounds"
    assert "550.0 A" in alert_tampered.details

    # Low temperature anomaly test: pack temperature -25 C is below -20 C safe minimum
    msg_freezing = encode_bms_status(90.0, 380.0, 50.0, -25.0, timestamp=2.200)
    alert_freezing = detector.evaluate_message(msg_freezing)
    assert alert_freezing.is_alarm
    assert alert_freezing.alarm_type == "payload_out_of_bounds"
    assert "-25.0 C" in alert_freezing.details


def test_can_detector_detects_replay_attack() -> None:
    detector = CANSecurityDetector(expected_period_s=0.100, timing_tolerance_s=0.030)

    # Record 3 frames
    buffer = [
        encode_bms_status(current_demand_a=60.0, voltage_limit_v=380.0, soc_percent=40.0, pack_temp_c=22.0, timestamp=t)
        for t in [1.0, 1.1, 1.2]
    ]

    # Prime detector with frame 0
    detector.evaluate_message(buffer[0])

    # Replay attack: frames injected with 15 ms burst spacing (violates 100 ms period)
    replayed = inject_replay_attack(buffer[1:], replay_start_time=1.015, burst_period_s=0.015)

    alert_replay = detector.evaluate_message(replayed[0])
    assert alert_replay.is_alarm
    assert alert_replay.alarm_type == "timing_anomaly"
    assert "Burst / Replay" in alert_replay.details


def test_python_can_virtual_bus_integration() -> None:
    """Verify end-to-end communication over python-can virtual bus."""
    # Use in-memory virtual bus for cross-platform CI / macOS compatibility
    bus_tx = can.Bus(interface="virtual", channel="vcan_test")
    bus_rx = can.Bus(interface="virtual", channel="vcan_test")

    detector = CANSecurityDetector()

    msg_tx = encode_bms_status(
        current_demand_a=110.0,
        voltage_limit_v=400.0,
        soc_percent=72.0,
        pack_temp_c=30.0,
        timestamp=5.000,
    )
    bus_tx.send(msg_tx)

    msg_rx = bus_rx.recv(timeout=1.0)
    assert msg_rx is not None
    assert msg_rx.arbitration_id == CAN_ID_BMS_STATUS

    alert = detector.evaluate_message(msg_rx)
    assert not alert.is_alarm

    bus_tx.shutdown()
    bus_rx.shutdown()


class MockReadbackEVSE(EVSEInterface):
    def __init__(self, readback_value: float | Exception) -> None:
        self.readback_value = readback_value
        self.dispatched_setpoint: float | None = None

    def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
        self.dispatched_setpoint = power_kw

    def read_session_kw(self, ev_id: str) -> float:
        if isinstance(self.readback_value, Exception):
            raise self.readback_value
        return self.readback_value


class MockFailingSetpointEVSE(EVSEInterface):
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
        raise self.exc

    def read_session_kw(self, ev_id: str) -> float:
        return 0.0


def test_can_detector_failsafe_rejects_nan_and_infinite_readback() -> None:
    """EVSE readback of NaN or inf must never produce actuation_status='confirmed_zero'."""
    anomalous_msg = can.Message(
        arbitration_id=CAN_ID_BMS_STATUS,
        data=b"\x00" * 4,  # Malformed DLC
        is_extended_id=False,
        timestamp=1.0,
    )

    # 1. NaN readback must result in 'unverified', never 'confirmed_zero'
    evse_nan = MockReadbackEVSE(float("nan"))
    det_nan = CANSecurityDetector(evse=evse_nan, ev_id="EV_NAN")
    alert_nan = det_nan.evaluate_message(anomalous_msg)
    assert alert_nan.is_alarm is True
    assert alert_nan.actuation_status != "confirmed_zero"
    assert alert_nan.actuation_status == "unverified"

    # 2. Positive infinity readback must result in 'unverified'
    evse_inf = MockReadbackEVSE(float("inf"))
    det_inf = CANSecurityDetector(evse=evse_inf, ev_id="EV_INF")
    alert_inf = det_inf.evaluate_message(anomalous_msg)
    assert alert_inf.is_alarm is True
    assert alert_inf.actuation_status != "confirmed_zero"
    assert alert_inf.actuation_status == "unverified"

    # 3. Negative infinity readback must result in 'unverified'
    evse_neginf = MockReadbackEVSE(float("-inf"))
    det_neginf = CANSecurityDetector(evse=evse_neginf, ev_id="EV_NEGINF")
    alert_neginf = det_neginf.evaluate_message(anomalous_msg)
    assert alert_neginf.is_alarm is True
    assert alert_neginf.actuation_status != "confirmed_zero"
    assert alert_neginf.actuation_status == "unverified"

    # 4. Readback query exception must result in 'failed'
    evse_err = MockReadbackEVSE(RuntimeError("I2C read failure"))
    det_err = CANSecurityDetector(evse=evse_err, ev_id="EV_ERR")
    alert_err = det_err.evaluate_message(anomalous_msg)
    assert alert_err.is_alarm is True
    assert alert_err.actuation_status == "failed"

    # 5. Setpoint dispatch exception must result in 'failed'
    evse_cmd_err = MockFailingSetpointEVSE(RuntimeError("CAN command bus dropped"))
    det_cmd_err = CANSecurityDetector(evse=evse_cmd_err, ev_id="EV_CMD_ERR")
    alert_cmd_err = det_cmd_err.evaluate_message(anomalous_msg)
    assert alert_cmd_err.is_alarm is True
    assert alert_cmd_err.actuation_status == "failed"

    # 6. Finite zero readback within tolerance results in 'confirmed_zero'
    evse_zero = MockReadbackEVSE(0.0)
    det_zero = CANSecurityDetector(evse=evse_zero, ev_id="EV_ZERO")
    alert_zero = det_zero.evaluate_message(anomalous_msg)
    assert alert_zero.is_alarm is True
    assert alert_zero.actuation_status == "confirmed_zero"
