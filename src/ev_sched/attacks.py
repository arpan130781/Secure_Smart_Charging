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

import numpy as np
from typing import Any, Dict, Optional, Sequence, List
from ev_sched.hal import MeterInterface, EVSEInterface


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


class AdaptiveAttackMeter(MeterInterface):
    """Wraps MeterInterface with an adaptive, model-cognizant FDI attack.

    The attacker knows that the Energy Management System (EMS) monitors the
    energy-balance innovation residual:
        r(t) = reported_kw(t) - (P_base_forecast_kw[t] + sum(P_ev_cmd))

    To remain undetectable to Detector 1 (residual check / CUSUM), the adaptive
    attacker clamps the reported telemetry to the expected balance plus small noise:
        reported_kw(t) = P_base_forecast_kw[t] + ev_power_kw(t) + noise
    
    When an unmodeled load surge occurs physically (e.g. true base load increases
    by delta_surge_kw), the true transformer load exceeds capacity:
        P_true(t) = P_base_forecast_kw[t] + delta_surge_kw + sum(P_ev) > P_feeder_max
    Yet the residual remains zero-mean noise r(t) ~ N(0, sigma^2). Detector 1
    observes zero anomaly, resulting in zero alarms and undetected physical overload.

    Parameters
    ----------
    inner_meter : MeterInterface
        The physical or simulated meter being wrapped.
    start_min : int
        Simulation elapsed time when attack begins in minutes [minutes].
    duration_min : int
        Duration of the attack window in minutes [minutes].
    base_forecast_profile_kw : Sequence[float]
        EMS base load forecast known to the adaptive attacker in kilowatts [kW].
    simulator : Optional[Any], default None
        Simulator reference to observe active EV power (or EVSE interface).
    time_step_min : int, default 15
        Time step resolution in minutes [minutes].
    noise_std_kw : float, default 0.2
        Residual noise standard deviation injected to mimic legitimate telemetry [kW].
    seed : int, default 42
        Random seed for reproducible noise generation.
    """

    def __init__(
        self,
        inner_meter: MeterInterface,
        start_min: int,
        duration_min: int,
        base_forecast_profile_kw: Sequence[float],
        simulator: Optional[Any] = None,
        time_step_min: int = 15,
        noise_std_kw: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.inner_meter = inner_meter
        self.start_min = start_min
        self.duration_min = duration_min
        self.end_min = start_min + duration_min
        self.base_forecast_profile_kw = list(base_forecast_profile_kw)
        self.simulator = simulator
        self.time_step_min = time_step_min
        self.noise_std_kw = noise_std_kw
        self.rng = np.random.default_rng(seed)

    def is_attack_active(self, t_min: int) -> bool:
        """Return True if cyber-attack is actively injecting false data at t_min."""
        return self.start_min <= t_min < self.end_min

    def read_feeder_kw(self, t_min: int) -> float:
        """[MEASURED] Return adaptive feeder reading in kilowatts [kW]."""
        true_kw = self.inner_meter.read_feeder_kw(t_min)
        if not self.is_attack_active(t_min):
            return true_kw

        step = t_min // self.time_step_min
        forecast_base = (
            self.base_forecast_profile_kw[step]
            if step < len(self.base_forecast_profile_kw)
            else self.base_forecast_profile_kw[-1]
        )

        # Inspect current active EV power from simulator if available
        ev_kw = 0.0
        if self.simulator is not None and hasattr(self.simulator, "get_total_ev_power_kw"):
            ev_kw = float(self.simulator.get_total_ev_power_kw())

        # Synthesize reported reading perfectly matching expected balance + noise
        noise = float(self.rng.normal(0.0, self.noise_std_kw))
        reported = max(0.0, forecast_base + ev_kw + noise)
        return reported


def create_attacked_meter(inner_meter: MeterInterface, config: Dict[str, Any]) -> MeterInterface:
    """Factory creating an Attacked MeterInterface wrapper from a configuration dictionary.

    Parameters
    ----------
    inner_meter : MeterInterface
        Legitimate meter to wrap.
    config : Dict[str, Any]
        Configuration dictionary containing:
        - attack_type: 'constant_bias', 'ramp', 'replay', or 'adaptive'
        - start_min: start time in elapsed minutes [minutes]
        - duration_min: duration in minutes [minutes]
        - magnitude_kw: bias magnitude in kilowatts [kW] (for bias and ramp)
        - replay_source_start_min: historical window start in minutes (for replay)
        - base_forecast_profile_kw: forecast profile for adaptive attack
        - simulator: simulator reference for adaptive attack

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
    elif atk_type == "adaptive":
        base_forecast = config.get("base_forecast_profile_kw", [30.0] * 96)
        sim = config.get("simulator", None)
        return AdaptiveAttackMeter(
            inner_meter=inner_meter,
            start_min=start_min,
            duration_min=duration_min,
            base_forecast_profile_kw=base_forecast,
            simulator=sim,
        )
    else:
        raise ValueError(f"Unknown attack type: {atk_type}")
