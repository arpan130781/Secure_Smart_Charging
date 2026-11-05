"""CAN Protocol definitions for EV Battery Management System (BMS) to EVSE Charger.

Conforms to standard EV DC fast charging conventions (ISO 15118 / DIN 70121 / SAE J1939).
CAN ID 0x101: BMS_Status message transmitted at 100 ms period (10 Hz).
"""

from dataclasses import dataclass
import struct
from typing import Optional
import can


@dataclass(frozen=True)
class BMSStatusMessage:
    """Decoded physical signals from BMS CAN frame 0x101.

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
    timestamp : float
        Message reception timestamp in seconds [s].
    """

    current_demand_a: float
    voltage_limit_v: float
    soc_percent: float
    pack_temp_c: float
    timestamp: float = 0.0


CAN_ID_BMS_STATUS = 0x101


def encode_bms_status(
    current_demand_a: float,
    voltage_limit_v: float,
    soc_percent: float,
    pack_temp_c: float,
    timestamp: float = 0.0,
) -> can.Message:
    """Encode physical BMS telemetry into an 8-byte CAN frame.

    Byte layout (Little Endian):
    - Bytes 0-1: current_demand_a (uint16, scale 0.1 A, range 0-6553.5 A)
    - Bytes 2-3: voltage_limit_v  (uint16, scale 0.1 V, range 0-6553.5 V)
    - Byte 4   : soc_percent      (uint8,  scale 0.5 %, range 0-127.5 %)
    - Byte 5   : pack_temp_c      (int8,   offset -40 C, range -40 to 87 C)
    - Bytes 6-7: reserved (0x00, 0x00)
    """
    raw_current = int(round(max(0.0, current_demand_a) / 0.1)) & 0xFFFF
    raw_voltage = int(round(max(0.0, voltage_limit_v) / 0.1)) & 0xFFFF
    raw_soc = int(round(max(0.0, min(100.0, soc_percent)) / 0.5)) & 0xFF
    raw_temp = int(round(pack_temp_c + 40.0)) & 0xFF

    payload = struct.pack("<HHBBxx", raw_current, raw_voltage, raw_soc, raw_temp)
    return can.Message(
        arbitration_id=CAN_ID_BMS_STATUS,
        data=payload,
        is_extended_id=False,
        timestamp=timestamp,
    )


def decode_bms_status(msg: can.Message) -> Optional[BMSStatusMessage]:
    """Decode an 8-byte CAN frame into a BMSStatusMessage dataclass."""
    if msg.arbitration_id != CAN_ID_BMS_STATUS or len(msg.data) < 6:
        return None

    raw_current, raw_voltage, raw_soc, raw_temp = struct.unpack("<HHBB", msg.data[:6])
    current_a = raw_current * 0.1
    voltage_v = raw_voltage * 0.1
    soc = raw_soc * 0.5
    temp_c = raw_temp - 40.0

    return BMSStatusMessage(
        current_demand_a=float(current_a),
        voltage_limit_v=float(voltage_v),
        soc_percent=float(soc),
        pack_temp_c=float(temp_c),
        timestamp=float(msg.timestamp),
    )
