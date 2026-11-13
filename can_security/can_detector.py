"""Intrusion Detection System (IDS) for EV CAN Bus communications.

RESEARCH & SIMULATED PROTOCOL NOTICE:
This detector operates on simulated project-specific CAN profiles:
1. CAN 2.0 Demonstration Profile (8 bytes): 1-byte HMAC tag truncated for demonstration ($2^{-8} = 256$ states).
2. CAN FD Hardened Profile (16 bytes): 8-byte (64-bit) HMAC-SHA256 tag ($2^{-64}$ collision resistance).

Security Boundary & Architectural Assumptions:
- Neither profile implements or claims compliance with official interoperable automotive standards
  (e.g., AUTOSAR SecOC or ISO 15118-20 PLC/TLS).
- Pre-shared symmetric keys are assumed; dynamic key distribution, cryptographic certificates,
  and asymmetric key provisioning are outside the current simulation scope.
- Replay protection across receiver restarts incorporates the receiver's boot epoch into the HMAC
  digest, assuming out-of-band authenticated epoch synchronization (e.g. ISO 15118 TLS handshake
  or SecOC Sync Message broadcast) between transmitter and receiver.
- State persistence uses POSIX advisory file locking (fcntl.flock) and monotonic update enforcement to
  prevent concurrent process races and software rollback on local storage; full physical tamper resistance
  in vehicular ECUs requires hardware-rooted monotonic counters (TPM, HSM, or secure element).

Fail-Safe Control Semantics:
- Advisory Mode (default, evse=None): Returns diagnostic CANAnomalyAlert with recommended
  safe_current_setpoint_a = 0.0 A and actuation_status='not_connected'.
- Autonomous Actuation Mode (evse is provided): Dispatches 0.0 kW setpoint to the specified
  EVSE session via EVSEInterface.set_setpoint_kw(), catches hardware exceptions, and verifies
  actuation via submeter readback (EVSEInterface.read_session_kw()), reporting explicit
  actuation_status ('confirmed_zero', 'unverified', or 'failed') and readback_power_kw.

Performs:
1. Frame Format & DLC Validation: Rejects remote frames (RTR), error frames, extended 29-bit CAN IDs, and payloads != 8 bytes.
2. Timing Check: Evaluates inter-arrival interval against nominal 100 ms period (+/- tolerance).
3. Authenticated Freshness Check: Enforces rolling sequence counter progression and HMAC tag verification with receiver boot epoch.
4. Payload Range Check: Enforces strict physical pack bounds on current, voltage, SOC, and temperature.
5. Fail-Safe Actuation: Dispatches 0.0 A / 0.0 kW setpoint upon anomaly detection with readback verification.
"""

import fcntl
import logging
import math
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import can

from can_security.bms_protocol import (
    CAN_ID_BMS_STATUS,
    BMSStatusMessage,
    compute_auth_tag,
    compute_auth_tag_64bit,
    decode_bms_status,
    decode_bms_status_canfd,
)

logger = logging.getLogger(__name__)
_STATE_LOCK = threading.RLock()

if TYPE_CHECKING:
    from ev_sched.hal import EVSEInterface


@dataclass
class CANAnomalyAlert:
    """Diagnostic outcome returned for each received CAN frame.

    Parameters
    ----------
    is_alarm : bool
        True if the frame violates framing, timing, freshness, or physical bounds.
    alarm_type : Optional[str]
        Category of violation: 'invalid_frame_format', 'error_frame_detected',
        'payload_malformed', 'timing_anomaly', 'freshness_anomaly', 'auth_failure',
        'payload_out_of_bounds', or None.
    details : str
        Human-readable explanation of why the message was flagged.
    safe_current_setpoint_a : float, default 0.0
        Fail-safe charging current setpoint [A] (0.0 A when alarmed).
    status : Optional[BMSStatusMessage], default None
        Decoded BMS status message if decoding succeeded.
    actuation_status : str, default 'no_alarm'
        Fail-safe actuation status:
        - 'no_alarm' : Normal operation, no fail-safe triggered.
        - 'not_connected' : Anomaly detected, but no EVSE interface is connected (advisory mode).
        - 'confirmed_zero' : Fail-safe 0.0 kW setpoint dispatched and verified via submeter readback.
        - 'unverified' : Setpoint dispatched, but submeter readback indicates charging > 0.05 kW.
        - 'failed' : Setpoint dispatch or readback query raised a hardware exception.
    readback_power_kw : Optional[float], default None
        Measured active power from EVSE submeter during fail-safe readback verification [kW].
    """

    is_alarm: bool
    alarm_type: str | None
    details: str
    safe_current_setpoint_a: float = 0.0
    status: BMSStatusMessage | None = None
    actuation_status: str = "no_alarm"
    readback_power_kw: float | None = None


class CANSecurityDetector:
    """Real-time CAN bus intrusion detector for EV BMS communications.

    DEMONSTRATION & SIMULATED PROTOCOL NOTICE:
    This detector operates on simulated project-specific CAN protocols:
    1. CAN 2.0 Demonstration Profile (8 bytes): 1-byte HMAC tag truncated for demonstration ($2^{-8} = 256$ states).
    2. CAN FD Profile (16 bytes): 8-byte (64-bit) HMAC-SHA256 tag ($2^{-64}$ states).
    This is a research simulation inspired by SecOC principles (ISO 21434 / AUTOSAR), but does NOT implement
    or claim compliance with an interoperable AUTOSAR SecOC profile.

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
    min_temp_c : float, default -20.0
        Minimum safe operational battery temperature in degrees Celsius [deg C].
    max_temp_c : float, default 65.0
        Maximum safe operational battery temperature in degrees Celsius [deg C].
    enforce_freshness : bool, default False
        Whether to reject stale or non-monotonic sequence counters.
    enforce_auth : bool, default False
        Whether to require cryptographic authentication. When True, auth_key must be non-empty.
    auth_key : bytes, default b""
        Shared secret key for verifying truncated HMAC authentication tag.
    max_counter_jump : int, default 16
        Maximum allowable forward jump in rolling counter before flagging anomaly.
    boot_epoch : int, default 0
        Receiver-validated session or boot epoch. Incremented across detector restarts.
    evse : Optional[EVSEInterface], default None
        Optional EVSE control interface. When provided, anomalies actively actuate 0.0 kW setpoint.
    ev_id : Optional[str], default None
        Target EV session identifier for EVSE fail-safe actuation.
    secure_mode : bool, default False
        Convenience preset for high-security configurations. Enforces freshness and auth,
        requiring a non-empty auth_key.
    """

    def __init__(
        self,
        expected_period_s: float = 0.100,
        timing_tolerance_s: float = 0.040,
        max_allowable_current_a: float = 200.0,
        min_voltage_v: float = 200.0,
        max_voltage_v: float = 450.0,
        min_temp_c: float = -20.0,
        max_temp_c: float = 65.0,
        enforce_freshness: bool = False,
        enforce_auth: bool = False,
        auth_key: bytes = b"",
        max_counter_jump: int = 16,
        boot_epoch: int = 0,
        evse: "EVSEInterface | None" = None,
        ev_id: str | None = None,
        secure_mode: bool = False,
        production_mode: bool = False,
        state_file: Path | str | None = None,
    ) -> None:
        self.expected_period_s = expected_period_s
        self.timing_tolerance_s = timing_tolerance_s
        self.max_allowable_current_a = max_allowable_current_a
        self.min_voltage_v = min_voltage_v
        self.max_voltage_v = max_voltage_v
        self.min_temp_c = min_temp_c
        self.max_temp_c = max_temp_c
        self.max_counter_jump = max_counter_jump
        self.evse = evse
        self.ev_id = ev_id
        self.secure_mode = secure_mode
        self.production_mode = production_mode

        if production_mode:
            self.enforce_freshness = True
            self.enforce_auth = True
            if len(auth_key) < 16:
                raise ValueError(
                    f"Production-secure mode requires a cryptographic key of at least 16 bytes (128 bits), got {len(auth_key)} bytes."
                )
            if state_file is None:
                raise ValueError(
                    "Production mode requires a durable, persistent state_file for trustworthy freshness state across restarts."
                )
        elif secure_mode:
            self.enforce_freshness = True
            self.enforce_auth = True
            if not auth_key:
                raise ValueError("Secure configuration (secure_mode=True) requires a non-empty cryptographic auth_key.")
        else:
            self.enforce_freshness = enforce_freshness
            self.enforce_auth = enforce_auth or bool(auth_key)
            if self.enforce_auth and not auth_key:
                raise ValueError("Authentication is enforced (enforce_auth=True) but auth_key is empty.")

        # Persistence of boot epoch across process restarts
        self.state_file = Path(state_file) if state_file is not None else None
        if self.state_file is not None:
            self.boot_epoch = self._load_and_increment_epoch()
        else:
            self.boot_epoch = boot_epoch

        self.auth_key = auth_key
        self.last_msg_timestamp: float | None = None
        self.last_counter: int | None = None

    def get_active_epoch(self) -> int:
        """Return receiver's current active boot epoch.

        Transmitters synchronize with this epoch out-of-band (e.g. via ISO 15118 TLS session
        negotiation or SecOC Sync Message) prior to computing HMAC authentication tags.
        """
        return self.boot_epoch

    def reset(self, new_epoch: int | None = None) -> None:
        """Reset internal timing, counter history, and advance receiver boot epoch.

        Parameters
        ----------
        new_epoch : Optional[int], default None
            Specific receiver session epoch to set. If provided, it must be strictly
            greater than current boot_epoch to prevent rollback attacks.
            If None, increments current epoch by 1.
        """
        self.last_msg_timestamp = None
        self.last_counter = None
        if new_epoch is not None:
            target_epoch = int(new_epoch)
            if target_epoch <= self.boot_epoch:
                raise ValueError(
                    f"Epoch rollback rejected: new_epoch ({target_epoch}) must be strictly greater "
                    f"than current epoch ({self.boot_epoch})"
                )
            self.boot_epoch = target_epoch
        else:
            self.boot_epoch += 1
        if self.state_file is not None:
            self._save_epoch(self.boot_epoch)

    @staticmethod
    def _get_lock_path(state_file: Path | str) -> Path:
        p = Path(state_file)
        return p.with_name(f".{p.name}.lock")

    @classmethod
    @contextmanager
    def _file_lock(cls, lock_path: Path):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with _STATE_LOCK:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    @staticmethod
    def initialize_state_file(path: Path | str, initial_epoch: int = 1) -> None:
        """Explicitly provision or initialize a trusted epoch state file.

        Acquires a stable lock before inspecting the state file and creates the file
        exclusively (O_CREAT | O_EXCL). Fails if the file already exists, preserving
        existing epoch state without truncating or modifying file contents.

        Parameters
        ----------
        path : Path | str
            Target file path for the epoch state file.
        initial_epoch : int, default 1
            Starting epoch value (must be >= 1).
        """
        p = Path(path)
        if initial_epoch < 1:
            raise ValueError(f"Initial epoch must be >= 1, got {initial_epoch}")
        p.parent.mkdir(parents=True, exist_ok=True)
        lock_path = CANSecurityDetector._get_lock_path(p)
        with CANSecurityDetector._file_lock(lock_path):
            if p.exists():
                raise FileExistsError(
                    f"State file '{p}' already exists; cannot overwrite or reset existing epoch state."
                )
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                with open(fd, "w", encoding="utf-8", closefd=False) as f:
                    f.write(f"{int(initial_epoch)}\n")
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                os.close(fd)

    def _load_and_increment_epoch(self) -> int:
        """Atomically read, validate, increment, and persist the next monotonic epoch.

        Acquires a stable lock before checking file existence, serializing concurrent
        first-time initialization and subsequent restarts to guarantee duplicate
        epochs can never be assigned.
        """
        assert self.state_file is not None
        p = self.state_file
        lock_path = self._get_lock_path(p)
        with self._file_lock(lock_path):
            if not p.exists():
                if self.production_mode:
                    raise FileNotFoundError(
                        f"Production state file missing: '{p}'. "
                        f"Production mode fails closed without persistent, trusted epoch state."
                    )
                # First-time initialization in non-production mode:
                # Atomically create state file with baseline epoch 1
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                try:
                    with open(fd, "w", encoding="utf-8", closefd=False) as f:
                        f.write("1\n")
                        f.flush()
                        os.fsync(f.fileno())
                finally:
                    os.close(fd)
                return 1

            try:
                with open(p, "r+", encoding="utf-8") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        content = f.read().strip()
                        if not content:
                            raise ValueError("State file is empty")
                        stored = int(content)
                        if stored < 1:
                            raise ValueError(f"Non-positive epoch in state file: {stored}")
                        epoch = stored + 1
                        f.seek(0)
                        f.truncate(0)
                        f.write(f"{epoch}\n")
                        f.flush()
                        os.fsync(f.fileno())
                        return epoch
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception as e:
                if self.production_mode:
                    raise RuntimeError(
                        f"Corrupt or unreadable epoch state file '{p}': {e}. "
                        f"Failing closed in production mode."
                    ) from e
                logger.warning("Could not read epoch state file %s: %s; starting at epoch 1", p, e)
                return 1

    def _save_epoch(self, epoch: int) -> None:
        """Atomically persist a new epoch, enforcing strict monotonic progression.

        Rejects any attempt to roll back or rewrite an epoch <= stored epoch.
        """
        assert self.state_file is not None
        p = self.state_file
        lock_path = self._get_lock_path(p)
        with self._file_lock(lock_path):
            if not p.exists():
                if self.production_mode:
                    raise FileNotFoundError(
                        f"Production state file missing: '{p}'."
                    )
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                try:
                    with open(fd, "w", encoding="utf-8", closefd=False) as f:
                        f.write(f"{int(epoch)}\n")
                        f.flush()
                        os.fsync(f.fileno())
                finally:
                    os.close(fd)
                return

            try:
                with open(p, "r+", encoding="utf-8") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        content = f.read().strip()
                        if content:
                            stored = int(content)
                            if epoch <= stored:
                                raise ValueError(
                                    f"Rollback rejected: new epoch {epoch} must be strictly greater "
                                    f"than persisted epoch {stored}"
                                )
                        f.seek(0)
                        f.truncate(0)
                        f.write(f"{epoch}\n")
                        f.flush()
                        os.fsync(f.fileno())
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception as e:
                if self.production_mode:
                    raise RuntimeError(
                        f"Cannot update epoch state file '{p}': {e}. "
                        f"Failing closed in production mode."
                    ) from e
                logger.warning("Could not update epoch state file %s: %s", p, e)

    def _actuate_failsafe_if_configured(self, alert: CANAnomalyAlert) -> bool:
        """Actuate fail-safe 0.0 kW setpoint on EVSEInterface if hardware control is connected.

        Catches dispatch exceptions from EVSEInterface.set_setpoint_kw(), performs readback
        verification via EVSEInterface.read_session_kw(), and updates alert.actuation_status
        and alert.readback_power_kw.

        Returns
        -------
        bool
            True if setpoint was dispatched and verified via readback (confirmed_zero); False otherwise.
        """
        if not alert.is_alarm:
            alert.actuation_status = "no_alarm"
            alert.readback_power_kw = None
            return False

        if self.evse is None or self.ev_id is None:
            alert.actuation_status = "not_connected"
            alert.readback_power_kw = None
            return False

        # Attempt to dispatch 0.0 kW setpoint to EVSE
        try:
            self.evse.set_setpoint_kw(self.ev_id, 0.0)
        except Exception as e:  # noqa: BLE001
            logger.error("EVSE fail-safe setpoint command failed for session %s: %s", self.ev_id, e)
            alert.actuation_status = "failed"
            alert.readback_power_kw = None
            return False

        # Attempt submeter readback verification
        try:
            delivered = float(self.evse.read_session_kw(self.ev_id))
            alert.readback_power_kw = delivered
            if not math.isfinite(delivered):
                logger.warning(
                    "EVSE fail-safe readback non-finite for session %s: %s",
                    self.ev_id,
                    delivered,
                )
                alert.actuation_status = "unverified"
                return False
            if abs(delivered) > 0.05:
                logger.warning(
                    "EVSE fail-safe readback mismatch for session %s: commanded 0.0 kW, measured %.2f kW",
                    self.ev_id,
                    delivered,
                )
                alert.actuation_status = "unverified"
                return False
            alert.actuation_status = "confirmed_zero"
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("EVSE fail-safe readback query failed for session %s: %s", self.ev_id, e)
            alert.actuation_status = "failed"
            alert.readback_power_kw = None
            return False

    def evaluate_message(self, msg: can.Message) -> CANAnomalyAlert:
        """Inspect a single CAN frame for framing, timing, freshness, and physical anomalies.

        State update policy:
        Timing and freshness state (last_msg_timestamp and last_counter) are updated ONLY
        upon successful validation of a completely valid frame. An invalid or anomalous frame
        NEVER corrupts detector state.

        Parameters
        ----------
        msg : can.Message
            Received CAN message frame.

        Returns
        -------
        CANAnomalyAlert
            Diagnostic outcome with fail-safe charging current setpoint.
        """
        # 1. Non-BMS message filtering
        # The CAN bus is shared by multiple ECUs. Messages with unrelated arbitration IDs
        # must be ignored without triggering alarms or actuating fail-safe on our EVSE!
        if msg.arbitration_id != CAN_ID_BMS_STATUS:
            return CANAnomalyAlert(
                is_alarm=False,
                alarm_type=None,
                details="Ignored non-BMS arbitration ID",
                safe_current_setpoint_a=0.0,
            )

        # 2. Framing and Format Validation for BMS Status frames (CAN_ID_BMS_STATUS)
        if getattr(msg, "is_remote_frame", False):
            alert = CANAnomalyAlert(
                is_alarm=True,
                alarm_type="invalid_frame_format",
                details="Remote Transmission Request (RTR) frame rejected; expected standard data frame",
                safe_current_setpoint_a=0.0,
            )
            self._actuate_failsafe_if_configured(alert)
            return alert

        if getattr(msg, "is_error_frame", False):
            alert = CANAnomalyAlert(
                is_alarm=True,
                alarm_type="error_frame_detected",
                details="CAN Error frame received; physical or signaling bus error detected",
                safe_current_setpoint_a=0.0,
            )
            self._actuate_failsafe_if_configured(alert)
            return alert

        if getattr(msg, "is_extended_id", False):
            alert = CANAnomalyAlert(
                is_alarm=True,
                alarm_type="invalid_frame_format",
                details="Extended 29-bit CAN identifier rejected; expected standard 11-bit framing",
                safe_current_setpoint_a=0.0,
            )
            self._actuate_failsafe_if_configured(alert)
            return alert

        # 3. Validate timestamp before using it
        try:
            current_time = float(msg.timestamp)
        except (TypeError, ValueError):
            alert = CANAnomalyAlert(
                is_alarm=True,
                alarm_type="invalid_timestamp",
                details="CAN frame has invalid non-numeric timestamp",
                safe_current_setpoint_a=0.0,
            )
            self._actuate_failsafe_if_configured(alert)
            return alert

        if not math.isfinite(current_time) or current_time < 0.0:
            alert = CANAnomalyAlert(
                is_alarm=True,
                alarm_type="invalid_timestamp",
                details=f"Invalid timestamp value: {current_time} (must be finite and non-negative)",
                safe_current_setpoint_a=0.0,
            )
            self._actuate_failsafe_if_configured(alert)
            return alert

        if self.last_msg_timestamp is not None and current_time < self.last_msg_timestamp:
            delta_t = current_time - self.last_msg_timestamp
            alert = CANAnomalyAlert(
                is_alarm=True,
                alarm_type="timing_anomaly",
                details=(
                    f"Backward timestamp detected: current {current_time:.3f} s < "
                    f"last {self.last_msg_timestamp:.3f} s (Delta={delta_t * 1000.0:.1f} ms)"
                ),
                safe_current_setpoint_a=0.0,
            )
            self._actuate_failsafe_if_configured(alert)
            return alert

        # 2. Production Mode vs Demonstration Protocol Checks
        is_fd = getattr(msg, "is_fd", False)
        if self.production_mode:
            # Production mode strictly prohibits standard 8-byte / CAN 2.0 fallback!
            if not is_fd or len(msg.data) == 8:
                alert = CANAnomalyAlert(
                    is_alarm=True,
                    alarm_type="insecure_protocol_fallback",
                    details=(
                        "Standard 8-byte CAN 2.0 frame rejected in production mode; "
                        "CAN FD with 64-bit MAC required to prevent 1-byte truncation attacks"
                    ),
                    safe_current_setpoint_a=0.0,
                )
                self._actuate_failsafe_if_configured(alert)
                return alert
            if len(msg.data) != 16:
                alert = CANAnomalyAlert(
                    is_alarm=True,
                    alarm_type="payload_malformed",
                    details=f"Invalid payload length: {len(msg.data)} bytes (expected 16 bytes for CAN FD BMS status)",
                    safe_current_setpoint_a=0.0,
                )
                self._actuate_failsafe_if_configured(alert)
                return alert
            status = decode_bms_status_canfd(msg)
        else:
            # Demonstration CAN 2.0 mode (or CAN FD if received)
            if is_fd and len(msg.data) == 16:
                status = decode_bms_status_canfd(msg)
            else:
                if len(msg.data) != 8:
                    alert = CANAnomalyAlert(
                        is_alarm=True,
                        alarm_type="payload_malformed",
                        details=f"Invalid payload length: {len(msg.data)} bytes (expected exactly 8 bytes)",
                        safe_current_setpoint_a=0.0,
                    )
                    self._actuate_failsafe_if_configured(alert)
                    return alert
                status = decode_bms_status(msg)

        if status is None:
            alert = CANAnomalyAlert(
                is_alarm=True,
                alarm_type="payload_malformed",
                details="Failed to decode payload structure",
                safe_current_setpoint_a=0.0,
            )
            self._actuate_failsafe_if_configured(alert)
            return alert

        # 3. Authenticated Freshness & Counter Progression Check
        delta_counter = 1
        if self.enforce_freshness and self.last_counter is not None:
            modulus = 65536 if status.is_fd else 256
            delta_counter = (status.counter - self.last_counter) % modulus
            if delta_counter == 0:
                alert = CANAnomalyAlert(
                    is_alarm=True,
                    alarm_type="freshness_anomaly",
                    details=f"Repeated frame detected: counter={status.counter} matches last_counter={self.last_counter}",
                    safe_current_setpoint_a=0.0,
                    status=status,
                )
                self._actuate_failsafe_if_configured(alert)
                return alert
            if delta_counter > self.max_counter_jump:
                alert = CANAnomalyAlert(
                    is_alarm=True,
                    alarm_type="freshness_anomaly",
                    details=(
                        f"Replay / Stale frame detected: counter={status.counter} "
                        f"not monotonically advancing from last_counter={self.last_counter} (delta={delta_counter})"
                    ),
                    safe_current_setpoint_a=0.0,
                    status=status,
                )
                self._actuate_failsafe_if_configured(alert)
                return alert

        # 4. Cryptographic Authentication Tag Verification
        if self.enforce_auth:
            if status.is_fd:
                expected_tag_bytes = compute_auth_tag_64bit(bytes(msg.data[:8]), self.auth_key, epoch=self.boot_epoch)
                if status.auth_tag_bytes != expected_tag_bytes:
                    alert = CANAnomalyAlert(
                        is_alarm=True,
                        alarm_type="auth_failure",
                        details=f"64-bit MAC verification failed (epoch={self.boot_epoch})",
                        safe_current_setpoint_a=0.0,
                        status=status,
                    )
                    self._actuate_failsafe_if_configured(alert)
                    return alert
            else:
                expected_tag = compute_auth_tag(bytes(msg.data[:7]), self.auth_key, epoch=self.boot_epoch)
                if status.auth_tag != expected_tag:
                    alert = CANAnomalyAlert(
                        is_alarm=True,
                        alarm_type="auth_failure",
                        details=f"MAC tag verification failed: received 0x{status.auth_tag:02X} != expected 0x{expected_tag:02X} (epoch={self.boot_epoch})",
                        safe_current_setpoint_a=0.0,
                        status=status,
                    )
                    self._actuate_failsafe_if_configured(alert)
                    return alert

        # 5. Timing Check (Inter-arrival interval delta_t)
        if self.last_msg_timestamp is not None:
            delta_t = current_time - self.last_msg_timestamp
            min_period = self.expected_period_s - self.timing_tolerance_s
            max_intervals = (
                max(1, delta_counter)
                if (self.enforce_freshness and self.last_counter is not None and delta_counter <= self.max_counter_jump)
                else max(1, round(delta_t / self.expected_period_s))
            )
            max_period = max_intervals * (self.expected_period_s + self.timing_tolerance_s)

            if delta_t < min_period:
                alert = CANAnomalyAlert(
                    is_alarm=True,
                    alarm_type="timing_anomaly",
                    details=f"Timing violation: delta_t = {delta_t * 1000.0:.1f} ms < {min_period * 1000.0:.1f} ms (Burst / Replay)",
                    safe_current_setpoint_a=0.0,
                    status=status,
                )
                self._actuate_failsafe_if_configured(alert)
                return alert
            elif delta_t > max_period:
                alert = CANAnomalyAlert(
                    is_alarm=True,
                    alarm_type="timing_anomaly",
                    details=f"Timing violation: delta_t = {delta_t * 1000.0:.1f} ms > {max_period * 1000.0:.1f} ms (Delay / Drop)",
                    safe_current_setpoint_a=0.0,
                    status=status,
                )
                self._actuate_failsafe_if_configured(alert)
                return alert

        # 6. Physical Payload Range Checks
        if status.current_demand_a < 0.0 or status.current_demand_a > self.max_allowable_current_a:
            alert = CANAnomalyAlert(
                is_alarm=True,
                alarm_type="payload_out_of_bounds",
                details=f"Current demand {status.current_demand_a:.1f} A exceeds limit [0.0, {self.max_allowable_current_a:.1f} A]",
                safe_current_setpoint_a=0.0,
                status=status,
            )
            self._actuate_failsafe_if_configured(alert)
            return alert

        if status.voltage_limit_v < self.min_voltage_v or status.voltage_limit_v > self.max_voltage_v:
            alert = CANAnomalyAlert(
                is_alarm=True,
                alarm_type="payload_out_of_bounds",
                details=f"Voltage limit {status.voltage_limit_v:.1f} V outside safe range [{self.min_voltage_v:.1f}, {self.max_voltage_v:.1f} V]",
                safe_current_setpoint_a=0.0,
                status=status,
            )
            self._actuate_failsafe_if_configured(alert)
            return alert

        if status.soc_percent < 0.0 or status.soc_percent > 100.0:
            alert = CANAnomalyAlert(
                is_alarm=True,
                alarm_type="payload_out_of_bounds",
                details=f"SOC {status.soc_percent:.1f}% outside physical bounds [0, 100%]",
                safe_current_setpoint_a=0.0,
                status=status,
            )
            self._actuate_failsafe_if_configured(alert)
            return alert

        if status.pack_temp_c < self.min_temp_c or status.pack_temp_c > self.max_temp_c:
            alert = CANAnomalyAlert(
                is_alarm=True,
                alarm_type="payload_out_of_bounds",
                details=(
                    f"Pack temperature {status.pack_temp_c:.1f} C outside safe operational range "
                    f"[{self.min_temp_c:.1f}, {self.max_temp_c:.1f} C]"
                ),
                safe_current_setpoint_a=0.0,
                status=status,
            )
            self._actuate_failsafe_if_configured(alert)
            return alert

        # 7. COMMIT STATE ONLY UPON SUCCESSFUL VALIDATION
        self.last_msg_timestamp = current_time
        self.last_counter = status.counter

        return CANAnomalyAlert(
            is_alarm=False,
            alarm_type=None,
            details="Message nominal",
            safe_current_setpoint_a=status.current_demand_a,
            status=status,
        )
