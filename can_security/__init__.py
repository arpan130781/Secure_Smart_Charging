"""can_security: CAN bus monitoring, simulation, attacks, and intrusion detection for EV charging."""

from can_security.bms_protocol import (
    CAN_ID_BMS_STATUS,
    BMSStatusMessage,
    compute_auth_tag,
    compute_auth_tag_64bit,
    decode_bms_status,
    decode_bms_status_canfd,
    encode_bms_status,
    encode_bms_status_canfd,
)
from can_security.can_attacks import (
    inject_nominal_timing_replay_attack,
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
    "compute_auth_tag",
    "compute_auth_tag_64bit",
    "decode_bms_status",
    "decode_bms_status_canfd",
    "encode_bms_status",
    "encode_bms_status_canfd",
    "inject_nominal_timing_replay_attack",
    "inject_replay_attack",
    "inject_value_tampering",
]
