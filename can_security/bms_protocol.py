"""Simulated custom CAN protocol for EV Battery Management System (BMS) to EVSE communications.

RESEARCH & SIMULATION NOTICE:
1. This is a project-specific simulated CAN protocol designed strictly for academic research
   and evaluation of physical bounds checking, timing IDS, and authenticated freshness.
2. It does NOT conform to ISO 15118 (which uses Power Line Communication with XML/EXI messages
   over IPv6), DIN 70121, or SAE J1939 (which uses 29-bit extended CAN identifiers).
3. Demonstration vs. Hardened Profiles:
   - CAN 2.0 Demonstration Profile (8 bytes): Uses an 11-bit ID `0x101`. Physical telemetry consumes
     6 bytes, leaving 2 bytes for security (1-byte counter + 1-byte truncated HMAC tag). A 1-byte
     tag provides only 256 unique states (1/256 collision probability), which is strictly for
     demonstration and insufficient for production automotive security.
   - CAN FD Hardened Profile (16 bytes): Extends payload to 16 bytes with a 16-bit sequence counter
     (0-65535) and an 8-byte (64-bit) truncated HMAC-SHA256 tag.
4. Transmitter-Receiver Epoch Synchronization Architecture:
   Replay protection across receiver restarts incorporates the receiver's boot epoch into the HMAC
   digest. In vehicular architectures, the transmitter learns the receiver's current epoch via an
   authenticated out-of-band handshake (e.g. ISO 15118 TLS session setup, SecOC Sync Message
   broadcast, or challenge-response). In this simulation, the transmitter is parameterized with
   the synchronized receiver epoch.

CAN ID 0x101: BMS_Status message transmitted at nominal 100 ms period (10 Hz)
using standard 11-bit CAN framing.
"""

import hashlib
import hmac
import struct
from dataclasses import dataclass

import can
import numpy as np


@dataclass(frozen=True)
class BMSStatusMessage:
    """Decoded physical signals and security fields from BMS CAN frame 0x101.

    Parameters
    ----------
    current_demand_a : float
        Requested battery charging current in Amperes [A].
    voltage_limit_v : float
        Maximum allowable pack terminal voltage in Volts [V].
    soc_percent : float
        Battery State of Charge in percent [%] (0.0 to 100.0).
    pack_temp_c : float
        Pack internal temperature in degrees Celsius [deg C].
    counter : int, default 0
        Rolling freshness sequence counter (0 to 255 for CAN 2.0, 0 to 65535 for CAN FD).
    auth_tag : int, default 0
        1-byte truncated HMAC authentication code (demonstration-only).
    timestamp : float, default 0.0
        Message reception timestamp in seconds [s].
    auth_tag_bytes : bytes, default b""
        Full authentication tag bytes (8 bytes for 64-bit CAN FD profile).
    is_fd : bool, default False
        True if received as a CAN FD frame.
    """

    current_demand_a: float
    voltage_limit_v: float
    soc_percent: float
    pack_temp_c: float
    counter: int = 0
    auth_tag: int = 0
    timestamp: float = 0.0
    auth_tag_bytes: bytes = b""
    is_fd: bool = False


CAN_ID_BMS_STATUS = 0x101


def compute_auth_tag(payload_prefix: bytes, key: bytes = b"", epoch: int = 0) -> int:
    """Compute 1-byte truncated HMAC-SHA256 authentication tag over payload bytes and epoch.

    Parameters
    ----------
    payload_prefix : bytes
        Data payload prefix to authenticate (bytes 0 to 6).
    key : bytes, default b""
        Shared cryptographic key. If empty, returns 0x00.
    epoch : int, default 0
        Receiver/transmitter synchronized session or boot epoch, bound into HMAC to
        prevent replay attacks across detector restarts.

    Returns
    -------
    int
        1-byte truncated HMAC tag (0 to 255).
    """
    if not key:
        return 0x00
    epoch_bytes = struct.pack("<I", epoch)
    digest = hmac.new(key, epoch_bytes + payload_prefix, hashlib.sha256).digest()
    return int(digest[0])


def encode_bms_status(
    current_demand_a: float,
    voltage_limit_v: float,
    soc_percent: float,
    pack_temp_c: float,
    timestamp: float = 0.0,
    counter: int = 0,
    auth_key: bytes = b"",
    is_extended_id: bool = False,
    epoch: int = 0,
) -> can.Message:
    """Encode physical BMS telemetry and freshness fields into an 8-byte standard CAN frame.

    Byte layout (Little Endian, 8 bytes exact):
    - Bytes 0-1: current_demand_a (uint16, scale 0.1 A, range 0-6553.5 A)
    - Bytes 2-3: voltage_limit_v  (uint16, scale 0.1 V, range 0-6553.5 V)
    - Byte 4   : soc_percent      (uint8,  scale 0.5 %, range 0-100.0 %)
    - Byte 5   : pack_temp_c      (int8,   offset -40 C, range -40 to 87 C)
    - Byte 6   : counter          (uint8,  rolling freshness counter 0-255)
    - Byte 7   : auth_tag         (uint8,  truncated HMAC authentication tag)
    """
    raw_current = int(np.clip(round(max(0.0, current_demand_a) / 0.1), 0, 0xFFFF))
    raw_voltage = int(np.clip(round(max(0.0, voltage_limit_v) / 0.1), 0, 0xFFFF))
    raw_soc = int(np.clip(round(max(0.0, min(100.0, soc_percent)) / 0.5), 0, 0xFF))
    raw_temp = int(np.clip(round(pack_temp_c + 40.0), 0, 0xFF))
    raw_counter = int(counter % 256)

    payload_prefix = struct.pack("<HHBBB", raw_current, raw_voltage, raw_soc, raw_temp, raw_counter)
    auth_tag = compute_auth_tag(payload_prefix, auth_key, epoch=epoch)
    payload = payload_prefix + struct.pack("B", auth_tag)

    return can.Message(
        arbitration_id=CAN_ID_BMS_STATUS,
        data=payload,
        is_extended_id=is_extended_id,
        timestamp=timestamp,
    )


def decode_bms_status(msg: can.Message) -> BMSStatusMessage | None:
    """Decode an 8-byte CAN frame into a BMSStatusMessage dataclass.

    Returns None if:
    - Frame is a Remote Transmission Request (is_remote_frame is True)
    - Frame is a CAN Error frame (is_error_frame is True)
    - Frame uses 29-bit extended ID (is_extended_id is True)
    - Arbitration ID != CAN_ID_BMS_STATUS (0x101)
    - Payload length != 8 bytes exact
    """
    if getattr(msg, "is_remote_frame", False) or getattr(msg, "is_error_frame", False):
        return None
    if getattr(msg, "is_extended_id", False) or msg.arbitration_id != CAN_ID_BMS_STATUS:
        return None
    if len(msg.data) != 8:
        return None

    raw_current, raw_voltage, raw_soc, raw_temp, counter, auth_tag = struct.unpack("<HHBBBB", msg.data)
    current_a = raw_current * 0.1
    voltage_v = raw_voltage * 0.1
    soc = raw_soc * 0.5
    temp_c = raw_temp - 40.0

    return BMSStatusMessage(
        current_demand_a=float(current_a),
        voltage_limit_v=float(voltage_v),
        soc_percent=float(soc),
        pack_temp_c=float(temp_c),
        counter=int(counter),
        auth_tag=int(auth_tag),
        timestamp=float(msg.timestamp),
        auth_tag_bytes=bytes([auth_tag]),
        is_fd=False,
    )


def compute_auth_tag_64bit(payload_prefix: bytes, key: bytes = b"", epoch: int = 0) -> bytes:
    """Compute 8-byte (64-bit) truncated HMAC-SHA256 authentication tag over payload bytes and epoch.

    Parameters
    ----------
    payload_prefix : bytes
        Data payload prefix to authenticate (bytes 0 to 7).
    key : bytes, default b""
        Shared cryptographic key. If empty, returns 8 zero bytes.
    epoch : int, default 0
        Receiver/transmitter synchronized session or boot epoch.

    Returns
    -------
    bytes
        8-byte (64-bit) truncated HMAC tag (project-specific simulated format inspired by SecOC principles).
    """
    if not key:
        return b"\x00" * 8
    epoch_bytes = struct.pack("<I", epoch)
    digest = hmac.new(key, epoch_bytes + payload_prefix, hashlib.sha256).digest()
    return digest[:8]


def encode_bms_status_canfd(
    current_demand_a: float,
    voltage_limit_v: float,
    soc_percent: float,
    pack_temp_c: float,
    timestamp: float = 0.0,
    counter: int = 0,
    auth_key: bytes = b"",
    epoch: int = 0,
) -> can.Message:
    """Encode physical BMS telemetry and 64-bit MAC into a 16-byte CAN FD frame.

    Byte layout (Little Endian, 16 bytes exact, CAN FD):
    - Bytes 0-1 : current_demand_a (uint16, scale 0.1 A, range 0-6553.5 A)
    - Bytes 2-3 : voltage_limit_v  (uint16, scale 0.1 V, range 0-6553.5 V)
    - Byte 4    : soc_percent      (uint8,  scale 0.5 %, range 0-100.0 %)
    - Byte 5    : pack_temp_c      (int8,   offset -40 C, range -40 to 87 C)
    - Bytes 6-7 : counter          (uint16, rolling freshness counter 0-65535)
    - Bytes 8-15: auth_tag         (8 bytes / 64-bit HMAC-SHA256 tag)

    Parameters
    ----------
    epoch : int, default 0
        Receiver session/boot epoch synchronized out-of-band (e.g. via ISO 15118 TLS
        handshake or SecOC Sync Message) prior to transmission.
    """
    raw_current = int(np.clip(round(max(0.0, current_demand_a) / 0.1), 0, 0xFFFF))
    raw_voltage = int(np.clip(round(max(0.0, voltage_limit_v) / 0.1), 0, 0xFFFF))
    raw_soc = int(np.clip(round(max(0.0, min(100.0, soc_percent)) / 0.5), 0, 0xFF))
    raw_temp = int(np.clip(round(pack_temp_c + 40.0), 0, 0xFF))
    raw_counter = int(counter % 65536)

    payload_prefix = struct.pack("<HHBBH", raw_current, raw_voltage, raw_soc, raw_temp, raw_counter)
    auth_tag_64 = compute_auth_tag_64bit(payload_prefix, auth_key, epoch=epoch)
    payload = payload_prefix + auth_tag_64

    return can.Message(
        arbitration_id=CAN_ID_BMS_STATUS,
        data=payload,
        is_extended_id=False,
        is_fd=True,
        timestamp=timestamp,
    )


def decode_bms_status_canfd(msg: can.Message) -> BMSStatusMessage | None:
    """Decode a 16-byte CAN FD frame into a BMSStatusMessage dataclass.

    Returns None if:
    - Frame is a Remote Transmission Request or Error frame
    - Frame uses 29-bit extended ID
    - Frame is not marked is_fd
    - Arbitration ID != CAN_ID_BMS_STATUS (0x101)
    - Payload length != 16 bytes exact
    """
    if getattr(msg, "is_remote_frame", False) or getattr(msg, "is_error_frame", False):
        return None
    if getattr(msg, "is_extended_id", False) or msg.arbitration_id != CAN_ID_BMS_STATUS:
        return None
    if not getattr(msg, "is_fd", False):
        return None
    if len(msg.data) != 16:
        return None

    raw_current, raw_voltage, raw_soc, raw_temp, counter, auth_tag_bytes = struct.unpack("<HHBBH8s", msg.data)
    current_a = raw_current * 0.1
    voltage_v = raw_voltage * 0.1
    soc = raw_soc * 0.5
    temp_c = raw_temp - 40.0

    return BMSStatusMessage(
        current_demand_a=float(current_a),
        voltage_limit_v=float(voltage_v),
        soc_percent=float(soc),
        pack_temp_c=float(temp_c),
        counter=int(counter),
        auth_tag=int(auth_tag_bytes[0]),
        timestamp=float(msg.timestamp),
        auth_tag_bytes=auth_tag_bytes,
        is_fd=True,
    )
