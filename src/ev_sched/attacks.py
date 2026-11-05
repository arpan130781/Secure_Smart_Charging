"""False Data Injection (FDI) cyber-attacks targeting grid feeder meters.

Threat Model:
-------------
An adversary has compromised the unencrypted / unauthenticated SCADA or AMI
telemetry link (e.g. Modbus/TCP or DNP3) between the substation feeder head meter
and the central EV energy management system.

The attack classes wrap a MeterInterface. From the perspective of the controller,
the interface behaves identically to a legitimate sensor, but returns manipulated
active power readings during the attack window.

Implemented Attacks:
--------------------
(a) Constant Bias: Subtracts delta_kw from the true reading (hiding feeder load).
(b) Ramp Attack: Bias delta_kw grows gradually from 0 to max_delta_kw to evade naive thresholds.
(c) Replay Attack: Replaces real-time telemetry with a recorded buffer of earlier readings.
"""

from typing import Any, Dict, Optional
from ev_sched.hal import MeterInterface


class ConstantBiasMeter(MeterInterface):
    """Wraps MeterInterface with an additive under-reporting bias attack.

    Parameters
    ----------
    inner_meter : MeterInterface
        The physical or simulated meter being wrapped.
    start_min : int
        Simulation elapsed time when attack begins in minutes [minutes].
    duration_min : int
        Duration of the attack window in minutes [minutes].
    delta_kw : float
        Power magnitude subtracted from the meter reading in kilowatts [kW].
    """

    def __init__(
        self,
        inner_meter: MeterInterface,
        start_min: int,
        duration_min: int,
        delta_kw: float,
    ) -> None:
        self.inner_meter = inner_meter
        self.start_min = start_min
        self.duration_min = duration_min
        self.end_min = start_min + duration_min
        self.delta_kw = abs(float(delta_kw))

    def is_attack_active(self, t_min: int) -> bool:
        """Return True if cyber-attack is actively injecting false data at t_min."""
        return self.start_min <= t_min < self.end_min

    def read_feeder_kw(self, t_min: int) -> float:
        """[MEASURED] Return attacked feeder active power reading in kilowatts [kW]."""
        true_kw = self.inner_meter.read_feeder_kw(t_min)
        if self.is_attack_active(t_min):
            # Under-report feeder load: reported = max(0.0, true - delta)
            return max(0.0, true_kw - self.delta_kw)
        return true_kw


class RampAttackMeter(MeterInterface):
    """Wraps MeterInterface with a gradual ramp drift attack.

    The bias grows linearly from 0 up to max_delta_kw across the attack duration.

    Parameters
    ----------
    inner_meter : MeterInterface
        The physical or simulated meter being wrapped.
    start_min : int
        Simulation elapsed time when attack begins in minutes [minutes].
    duration_min : int
        Duration of the attack window in minutes [minutes].
    max_delta_kw : float
        Maximum power subtraction ceiling reached at the end of the ramp in kilowatts [kW].
    """

    def __init__(
        self,
        inner_meter: MeterInterface,
        start_min: int,
        duration_min: int,
        max_delta_kw: float,
    ) -> None:
        self.inner_meter = inner_meter
        self.start_min = start_min
        self.duration_min = max(1, duration_min)
        self.end_min = start_min + duration_min
        self.max_delta_kw = abs(float(max_delta_kw))

    def is_attack_active(self, t_min: int) -> bool:
        """Return True if cyber-attack is actively injecting false data at t_min."""
        return self.start_min <= t_min < self.end_min

    def get_current_delta_kw(self, t_min: int) -> float:
        """Return the instantaneous injected bias at elapsed minute t_min [kW]."""
        if not self.is_attack_active(t_min):
            return 0.0
        elapsed = t_min - self.start_min
        progress = min(1.0, elapsed / self.duration_min)
        return progress * self.max_delta_kw

    def read_feeder_kw(self, t_min: int) -> float:
        """[MEASURED] Return attacked feeder active power reading in kilowatts [kW]."""
        true_kw = self.inner_meter.read_feeder_kw(t_min)
        if self.is_attack_active(t_min):
            delta = self.get_current_delta_kw(t_min)
            return max(0.0, true_kw - delta)
        return true_kw


class ReplayAttackMeter(MeterInterface):
    """Wraps MeterInterface with a replay attack.

    Buffers genuine readings during attack-free operation, and replays
    an earlier low-demand operational window during peak charging hours.

    Parameters
    ----------
    inner_meter : MeterInterface
        The physical or simulated meter being wrapped.
    start_min : int
        Simulation elapsed time when replay attack begins in minutes [minutes].
    duration_min : int
        Duration of the attack window in minutes [minutes].
    replay_source_start_min : int
        Start time of the recorded historical window to replay in minutes [minutes].
    """

    def __init__(
        self,
        inner_meter: MeterInterface,
        start_min: int,
        duration_min: int,
        replay_source_start_min: int,
    ) -> None:
        self.inner_meter = inner_meter
        self.start_min = start_min
        self.duration_min = duration_min
        self.end_min = start_min + duration_min
        self.replay_source_start_min = replay_source_start_min
        self._buffer: Dict[int, float] = {}

    def is_attack_active(self, t_min: int) -> bool:
        """Return True if cyber-attack is actively replaying telemetry at t_min."""
        return self.start_min <= t_min < self.end_min

    def read_feeder_kw(self, t_min: int) -> float:
        """[MEASURED] Return replayed or real-time feeder reading in kilowatts [kW]."""
        true_kw = self.inner_meter.read_feeder_kw(t_min)
        # Record into history buffer if not already stored
        if t_min not in self._buffer:
            self._buffer[t_min] = true_kw

        if self.is_attack_active(t_min):
            offset_min = t_min - self.start_min
            target_t = self.replay_source_start_min + offset_min
            return self._buffer.get(target_t, true_kw)

        return true_kw


def create_attacked_meter(inner_meter: MeterInterface, config: Dict[str, Any]) -> MeterInterface:
    """Factory creating an Attacked MeterInterface wrapper from a configuration dictionary.

    Parameters
    ----------
    inner_meter : MeterInterface
        Legitimate meter to wrap.
    config : Dict[str, Any]
        Configuration dictionary containing:
        - attack_type: 'constant_bias', 'ramp', or 'replay'
        - start_min: start time in elapsed minutes [minutes]
        - duration_min: duration in minutes [minutes]
        - magnitude_kw: bias magnitude in kilowatts [kW] (for bias and ramp)
        - replay_source_start_min: historical window start in minutes (for replay)

    Returns
    -------
    MeterInterface
        Wrapped meter instance.
    """
    atk_type = config.get("attack_type", "constant_bias")
    start_min = int(config.get("start_min", 480))
    duration_min = int(config.get("duration_min", 360))
    magnitude_kw = float(config.get("magnitude_kw", 35.0))

    if atk_type == "constant_bias":
        return ConstantBiasMeter(
            inner_meter=inner_meter,
            start_min=start_min,
            duration_min=duration_min,
            delta_kw=magnitude_kw,
        )
    elif atk_type == "ramp":
        return RampAttackMeter(
            inner_meter=inner_meter,
            start_min=start_min,
            duration_min=duration_min,
            max_delta_kw=magnitude_kw,
        )
    elif atk_type == "replay":
        replay_source = int(config.get("replay_source_start_min", 60))
        return ReplayAttackMeter(
            inner_meter=inner_meter,
            start_min=start_min,
            duration_min=duration_min,
            replay_source_start_min=replay_source,
        )
    else:
        raise ValueError(f"Unknown attack type: {atk_type}")
