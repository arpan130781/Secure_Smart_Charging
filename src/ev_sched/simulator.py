"""Grid and EV charging simulator with synthetic session generation and baseline charging.

Includes:
- Synthetic EV session generator: Poisson arrivals with a morning peak,
  lognormal parking durations, and truncated normal energy demands from YAML.
- Uncontrolled charging baseline: Every EV charges at max_rate_kw immediately
  upon arrival until its requested energy is delivered or it departs.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ev_sched.models import EVSession, Feeder


@dataclass
class SimulationResult:
    """Diagnostic outputs from a 24-hour simulation run.

    Parameters
    ----------
    step_times_min : List[int]
        Elapsed simulation time at each 15-minute step in minutes [minutes].
    step_times_hours : List[float]
        Elapsed simulation time at each step in hours [hours].
    base_load_kw : List[float]
        [SIMULATED] Background non-EV grid load in kilowatts [kW].
    ev_load_kw : List[float]
        [SIMULATED] Aggregate EV charging active power in kilowatts [kW].
    total_feeder_load_kw : List[float]
        [SIMULATED] Total active power flow on the substation feeder in kilowatts [kW].
    transformer_limit_kw : float
        [SIMULATED] Feeder transformer continuous capacity rating in kilowatts [kW].
    overload_kw : List[float]
        [SIMULATED] Instantaneous active power exceeding transformer capacity in kilowatts [kW].
    sessions : List[EVSession]
        Fleet of EV sessions with final ground-truth delivered energies.
    """

    step_times_min: list[int]
    step_times_hours: list[float]
    base_load_kw: list[float]
    ev_load_kw: list[float]
    total_feeder_load_kw: list[float]
    transformer_limit_kw: float
    overload_kw: list[float]
    sessions: list[EVSession]


def generate_synthetic_sessions(
    config: dict[str, Any],
    rng: np.random.Generator,
) -> list[EVSession]:
    """Generate synthetic EV charging sessions according to probabilistic distributions.

    Parameters
    ----------
    config : Dict[str, Any]
        Dictionary loaded from YAML containing generation parameters:
        - num_evs: target number of vehicles (typically 20 to 50)
        - arrival: morning peak parameters (peak_hour, std_hour)
        - duration: lognormal parameters (mean_log_min, sigma_log)
        - energy: truncated normal parameters (mean_kwh, std_kwh, min_kwh, max_kwh)
        - max_rate_kw: float or list of available charging ratings [kW]
    rng : np.random.Generator
        Seeded pseudo-random number generator for strict reproducibility.

    Returns
    -------
    List[EVSession]
        Chronologically sorted list of generated EVSession instances.
    """
    num_evs = int(config.get("num_evs", 35))
    if not (10 <= num_evs <= 100):
        # Bound between 10 and 100; user specified 20-50 as standard range
        num_evs = 35

    arrival_cfg = config.get("arrival", {})
    peak_min = float(arrival_cfg.get("peak_hour", 8.5)) * 60.0  # default 8:30 AM = 510 min
    std_min = float(arrival_cfg.get("std_hour", 1.5)) * 60.0    # default 1.5 hours = 90 min

    dur_cfg = config.get("duration", {})
    mean_log = float(dur_cfg.get("mean_log_min", 5.7))          # exp(5.7) ~ 300 min (5 hours)
    sigma_log = float(dur_cfg.get("sigma_log", 0.35))

    energy_cfg = config.get("energy", {})
    mean_kwh = float(energy_cfg.get("mean_kwh", 28.0))
    std_kwh = float(energy_cfg.get("std_kwh", 8.0))
    min_kwh = float(energy_cfg.get("min_kwh", 10.0))
    max_kwh = float(energy_cfg.get("max_kwh", 60.0))

    rate_opts = config.get("max_rate_kw", [7.2, 11.0])
    if isinstance(rate_opts, (int, float)):
        rate_opts = [float(rate_opts)]

    sessions: list[EVSession] = []

    # 1. Non-Homogeneous Poisson Arrival Process (NHPP) with morning peak:
    # Under classical Poisson point process theory, conditioned on N arrivals over [0, T],
    # arrival times are distributed as independent random draws from the normalized
    # intensity density f(t) = lambda(t) / int_0^T lambda(u) du.
    # Intensity lambda(t) models morning commuter arrivals centered at peak_min with spread std_min.
    time_slots_min = np.arange(0, 1440, 15)
    # Bounded operational arrival window between 06:00 (360 min) and 20:00 (1200 min)
    valid_window_mask = (time_slots_min >= 360) & (time_slots_min <= 1200)
    gaussian_intensity = np.exp(-0.5 * ((time_slots_min - peak_min) / max(1.0, std_min)) ** 2)
    poisson_intensity = np.where(valid_window_mask, gaussian_intensity, 0.0)
    intensity_sum = np.sum(poisson_intensity)
    arrival_probs = poisson_intensity / intensity_sum if intensity_sum > 0 else np.ones(96) / 96.0

    chosen_slots = rng.choice(len(time_slots_min), size=num_evs, p=arrival_probs)

    for i in range(num_evs):
        arr_min = int(time_slots_min[chosen_slots[i]])

        # 2. Parking duration: Lognormal distribution
        raw_dur = rng.lognormal(mean=mean_log, sigma=sigma_log)
        # Quantize to 15 minutes, minimum 60 minutes dwell time
        dur_min = max(60, int(round(raw_dur / 15.0) * 15))
        dep_min = min(1440, arr_min + dur_min)
        if dep_min <= arr_min:
            dep_min = min(1440, arr_min + 60)

        # 3. Energy demand: Truncated normal distribution
        raw_energy = rng.normal(loc=mean_kwh, scale=std_kwh)
        energy_kwh = float(np.clip(raw_energy, min_kwh, max_kwh))
        energy_kwh = round(energy_kwh, 2)

        # 4. Charger rating
        max_rate_kw = float(rng.choice(rate_opts))

        session = EVSession(
            arrival_min=arr_min,
            departure_min=dep_min,
            energy_req_kwh=energy_kwh,
            max_rate_kw=max_rate_kw,
            session_id=f"EV_{i+1:02d}",
            delivered_energy_kwh=0.0,
        )
        sessions.append(session)

    # Sort sessions chronologically by arrival time
    sessions.sort(key=lambda s: s.arrival_min)
    return sessions


def run_uncontrolled_charging(
    feeder: Feeder,
    sessions: list[EVSession],
    time_step_min: int = 15,
    num_steps: int = 96,
) -> SimulationResult:
    """Simulate uncontrolled 'dumb' charging baseline over 24 hours (96 steps).

    Every vehicle begins charging at its maximum continuous charging power (max_rate_kw)
    as soon as it arrives, until either its requested energy is fully satisfied or it departs.

    Parameters
    ----------
    feeder : Feeder
        Feeder parameters including transformer limit [kW] and base load profile [kW].
    sessions : List[EVSession]
        Fleet of EV sessions.
    time_step_min : int, default 15
        Interval duration in minutes [minutes].
    num_steps : int, default 96
        Total time steps (96 * 15 min = 1440 min = 24 hours).

    Returns
    -------
    SimulationResult
        Comprehensive time-series of base load, EV load, total feeder load, and overloads.
    """
    dt_hours = time_step_min / 60.0

    step_times_min: list[int] = []
    step_times_hours: list[float] = []
    base_load_kw: list[float] = []
    ev_load_kw: list[float] = []
    total_feeder_kw: list[float] = []
    overload_kw: list[float] = []

    # Reset delivered energy on all sessions
    for s in sessions:
        s.delivered_energy_kwh = 0.0

    # Ensure base load profile covers all steps
    base_profile = list(feeder.base_load_profile_kw)
    if len(base_profile) < num_steps:
        # Pad with last known value if needed
        last_val = base_profile[-1] if base_profile else 30.0
        base_profile.extend([last_val] * (num_steps - len(base_profile)))

    for step in range(num_steps):
        t_min = step * time_step_min
        step_times_min.append(t_min)
        step_times_hours.append(t_min / 60.0)

        step_base_kw = base_profile[step]
        base_load_kw.append(step_base_kw)

        # Calculate uncontrolled charging power across all sessions
        step_ev_power_kw = 0.0
        for s in sessions:
            if s.is_connected(t_min):
                rem_kwh = s.remaining_energy_req_kwh
                if rem_kwh > 1e-4:
                    # Max power this step bounded by remaining energy headroom
                    max_possible_kw = rem_kwh / dt_hours
                    power_kw = min(s.max_rate_kw, max_possible_kw)
                    energy_delivered_step = power_kw * dt_hours
                    s.delivered_energy_kwh += energy_delivered_step
                    step_ev_power_kw += power_kw

        ev_load_kw.append(step_ev_power_kw)
        step_total_kw = step_base_kw + step_ev_power_kw
        total_feeder_kw.append(step_total_kw)

        step_overload_kw = max(0.0, step_total_kw - feeder.transformer_limit_kw)
        overload_kw.append(step_overload_kw)

    return SimulationResult(
        step_times_min=step_times_min,
        step_times_hours=step_times_hours,
        base_load_kw=base_load_kw,
        ev_load_kw=ev_load_kw,
        total_feeder_load_kw=total_feeder_kw,
        transformer_limit_kw=feeder.transformer_limit_kw,
        overload_kw=overload_kw,
        sessions=sessions,
    )


class ChargingStationSimulator:
    """Stateful cyber-physical simulator wrapped by SimMeter and SimEVSE.

    Parameters
    ----------
    feeder : Feeder
        Feeder specifications (transformer capacity, base load profile).
    sessions : List[EVSession]
        List of EV charging sessions.
    time_step_min : int, default 15
        Simulation time-step resolution in minutes [minutes].
    rng : Optional[np.random.Generator]
        Seeded pseudo-random number generator for measurement noise.
    """

    def __init__(
        self,
        feeder: Feeder,
        sessions: list[EVSession],
        time_step_min: int = 15,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.feeder = feeder
        self.sessions = {s.session_id if s.session_id else f"EV_{i}": s for i, s in enumerate(sessions)}
        self.time_step_min = time_step_min
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.current_time_min: int = 0
        self.setpoints_kw: dict[str, float] = {sid: 0.0 for sid in self.sessions}

    def set_ev_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
        """[SIMULATED] Receive active power command from EVSEInterface."""
        if ev_id in self.sessions:
            self.setpoints_kw[ev_id] = max(0.0, float(power_kw))

    def get_session_power_kw(self, ev_id: str) -> float:
        """[SIMULATED & MEASURED] Get current physical charging power delivered to session."""
        if ev_id not in self.sessions:
            return 0.0
        session = self.sessions[ev_id]
        if not session.is_connected(self.current_time_min):
            return 0.0
        if session.remaining_energy_req_kwh <= 1e-4:
            return 0.0

        cmd_kw = self.setpoints_kw.get(ev_id, 0.0)
        dt_hours = self.time_step_min / 60.0
        max_energy_kw = session.remaining_energy_req_kwh / dt_hours
        return float(min(cmd_kw, session.max_rate_kw, max_energy_kw))

    def get_total_ev_power_kw(self) -> float:
        """[SIMULATED] Calculate sum of active power across all charging sessions."""
        return sum(self.get_session_power_kw(sid) for sid in self.sessions)

    def get_base_load_kw(self, t_min: int) -> float:
        """[SIMULATED] Lookup background non-EV grid load at elapsed time t_min."""
        step_idx = t_min // self.time_step_min
        profile = self.feeder.base_load_profile_kw
        if not profile:
            return 30.0
        if step_idx < len(profile):
            return float(profile[step_idx])
        return float(profile[-1])

    def get_feeder_power_kw(self, t_min: int) -> float:
        """[SIMULATED] Compute total feeder active power (base load + aggregate EV load)."""
        base_kw = self.get_base_load_kw(t_min)
        ev_kw = self.get_total_ev_power_kw()
        return base_kw + ev_kw

    def step(self) -> None:
        """[SIMULATED] Integrate physical battery energies and advance time by one interval."""
        dt_hours = self.time_step_min / 60.0
        for sid, session in self.sessions.items():
            power_kw = self.get_session_power_kw(sid)
            if power_kw > 0.0:
                energy_added = power_kw * dt_hours
                session.delivered_energy_kwh += energy_added

        self.current_time_min += self.time_step_min
