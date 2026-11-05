"""can_security: CAN bus monitoring, simulation, attacks, and intrusion detection for EV charging."""

from can_security.bms_protocol import (
    CAN_ID_BMS_STATUS,
    BMSStatusMessage,
    decode_bms_status,
    encode_bms_status,
)
from can_security.can_attacks import (
    inject_replay_attack,
    inject_value_tampering,
)
from can_security.can_detector import (
    CANAnomalyAlert,
    CANSecurityDetector,
)

__all__ = [
    "CAN_ID_BMS_STATUS",
    "BMSStatusMessage",
    "CANAnomalyAlert",
    "CANSecurityDetector",
    "decode_bms_status",
    "encode_bms_status",
    "inject_replay_attack",
    "inject_value_tampering",
]
