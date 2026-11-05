"""can_security: CAN bus monitoring, simulation, attacks, and intrusion detection for EV charging."""

from can_security.bms_protocol import (
    BMSStatusMessage,
    CAN_ID_BMS_STATUS,
    encode_bms_status,
    decode_bms_status,
)
from can_security.can_attacks import (
    inject_value_tampering,
    inject_replay_attack,
)
from can_security.can_detector import (
    CANAnomalyAlert,
    CANSecurityDetector,
)

__all__ = [
    "BMSStatusMessage",
    "CAN_ID_BMS_STATUS",
    "encode_bms_status",
    "decode_bms_status",
    "inject_value_tampering",
    "inject_replay_attack",
    "CANAnomalyAlert",
    "CANSecurityDetector",
]
