"""Intrusion Detection System (IDS) for EV CAN Bus communications.

Performs:
1. Timing Check: Evaluates inter-arrival message interval against nominal period (100 ms).
2. Payload Range Check: Enforces strict physical pack bounds on current, voltage, SOC, and temperature.
"""

from dataclasses import dataclass

import can

from can_security.bms_protocol import CAN_ID_BMS_STATUS, decode_bms_status


@dataclass(frozen=True)
class CANAnomalyAlert:
    """Diagnostic outcome returned for each received CAN frame.

    Parameters
    ----------
    is_alarm : bool
        True if the frame violates timing invariants or physical payload bounds.
    alarm_type : Optional[str]
        Category of violation: 'timing_anomaly', 'payload_out_of_bounds', or None.
    details : str
        Human-readable explanation of why the message was flagged.
    """

    is_alarm: bool
    alarm_type: str | None
    details: str


class CANSecurityDetector:
    """Real-time CAN bus intrusion detector for EV BMS communications.

    Parameters
    ----------
    expected_period_s : float, default 0.100
        Nominal transmission interval in seconds [s] (100 ms for 10 Hz BMS status).
    timing_tolerance_s : float, default 0.040
        Allowable timing jitter band [s] (+/- 40 ms).
    max_allowable_current_a : float, default 200.0
        Maximum physically safe charging current request in Amperes [A].
    min_voltage_v : float, default 200.0
        Minimum allowable battery terminal voltage in Volts [V].
    max_voltage_v : float, default 450.0
        Maximum allowable battery terminal voltage in Volts [V].
    max_temp_c : float, default 65.0
        Maximum safe operational battery temperature in degrees Celsius [deg C].
    """

    def __init__(
        self,
        expected_period_s: float = 0.100,
        timing_tolerance_s: float = 0.040,
        max_allowable_current_a: float = 200.0,
        min_voltage_v: float = 200.0,
        max_voltage_v: float = 450.0,
        max_temp_c: float = 65.0,
    ) -> None:
        self.expected_period_s = expected_period_s
        self.timing_tolerance_s = timing_tolerance_s
        self.max_allowable_current_a = max_allowable_current_a
        self.min_voltage_v = min_voltage_v
        self.max_voltage_v = max_voltage_v
        self.max_temp_c = max_temp_c

        self.last_msg_timestamp: float | None = None

    def reset(self) -> None:
        """Reset internal timing history."""
        self.last_msg_timestamp = None

    def evaluate_message(self, msg: can.Message) -> CANAnomalyAlert:
        """Inspect a single CAN frame for timing or physical value anomalies.

        Parameters
        ----------
        msg : can.Message
            Received CAN message frame.

        Returns
        -------
        CANAnomalyAlert
            Diagnostic outcome indicating whether the frame is safe or anomalous.
        """
        # Only evaluate BMS status messages
        if msg.arbitration_id != CAN_ID_BMS_STATUS:
            return CANAnomalyAlert(is_alarm=False, alarm_type=None, details="Ignored non-BMS arbitration ID")

        # 1. Timing Check (Inter-arrival interval delta_t)
        current_time = float(msg.timestamp)
        if self.last_msg_timestamp is not None:
            delta_t = current_time - self.last_msg_timestamp
            min_period = self.expected_period_s - self.timing_tolerance_s
            max_period = self.expected_period_s + self.timing_tolerance_s

            if delta_t < min_period:
                self.last_msg_timestamp = current_time
                return CANAnomalyAlert(
                    is_alarm=True,
                    alarm_type="timing_anomaly",
                    details=f"Timing violation: delta_t = {delta_t * 1000.0:.1f} ms < {min_period * 1000.0:.1f} ms (Burst / Replay)",
                )
            elif delta_t > max_period:
                self.last_msg_timestamp = current_time
                return CANAnomalyAlert(
                    is_alarm=True,
                    alarm_type="timing_anomaly",
                    details=f"Timing violation: delta_t = {delta_t * 1000.0:.1f} ms > {max_period * 1000.0:.1f} ms (Delay / Drop)",
                )

        self.last_msg_timestamp = current_time

        # 2. Payload Range Check
        status = decode_bms_status(msg)
        if status is None:
            return CANAnomalyAlert(
                is_alarm=True,
                alarm_type="payload_malformed",
                details="Failed to decode 8-byte payload structure",
            )

        # Check Current Demand
        if status.current_demand_a < 0.0 or status.current_demand_a > self.max_allowable_current_a:
            return CANAnomalyAlert(
                is_alarm=True,
                alarm_type="payload_out_of_bounds",
                details=f"Current demand {status.current_demand_a:.1f} A exceeds limit [0.0, {self.max_allowable_current_a:.1f} A]",
            )

        # Check Voltage Limit
        if status.voltage_limit_v < self.min_voltage_v or status.voltage_limit_v > self.max_voltage_v:
            return CANAnomalyAlert(
                is_alarm=True,
                alarm_type="payload_out_of_bounds",
                details=f"Voltage limit {status.voltage_limit_v:.1f} V outside safe range [{self.min_voltage_v:.1f}, {self.max_voltage_v:.1f} V]",
            )

        # Check State of Charge
        if status.soc_percent < 0.0 or status.soc_percent > 100.0:
            return CANAnomalyAlert(
                is_alarm=True,
                alarm_type="payload_out_of_bounds",
                details=f"SOC {status.soc_percent:.1f}% outside physical bounds [0, 100%]",
            )

        # Check Temperature
        if status.pack_temp_c > self.max_temp_c:
            return CANAnomalyAlert(
                is_alarm=True,
                alarm_type="payload_out_of_bounds",
                details=f"Pack temperature {status.pack_temp_c:.1f} C exceeds thermal threshold {self.max_temp_c:.1f} C",
            )

        return CANAnomalyAlert(is_alarm=False, alarm_type=None, details="Message nominal")
