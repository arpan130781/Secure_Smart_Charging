"""Grid and EV charging simulator with synthetic session generation and baseline charging.

Includes:
- Synthetic EV session generator: Poisson arrivals with a morning peak,
  lognormal parking durations, and truncated normal energy demands from YAML.
- Uncontrolled charging baseline: Every EV charges at max_rate_kw immediately
  upon arrival until its requested energy is delivered or it departs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
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

    step_times_min: List[int]
    step_times_hours: List[float]
    base_load_kw: List[float]
    ev_load_kw: List[float]
    total_feeder_load_kw: List[float]
    transformer_limit_kw: float
    overload_kw: List[float]
    sessions: List[EVSession]


def generate_synthetic_sessions(
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> List[EVSession]:
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

    sessions: List[EVSession] = []

    for i in range(num_evs):
        # 1. Arrival time: Gaussian cluster around morning peak (Poisson arrival intensity)
        # Bounded between 06:00 (360 min) and 20:00 (1200 min)
        raw_arr = rng.normal(loc=peak_min, scale=std_min)
        arr_min = int(np.clip(round(raw_arr), 360, 1200))
        # Quantize to 15-minute grid alignment
        arr_min = (arr_min // 15) * 15

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
    sessions: List[EVSession],
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

    step_times_min: List[int] = []
    step_times_hours: List[float] = []
    base_load_kw: List[float] = []
    ev_load_kw: List[float] = []
    total_feeder_kw: List[float] = []
    overload_kw: List[float] = []

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
