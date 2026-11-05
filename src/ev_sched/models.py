"""Domain models for EV charging, grid feeder limits, and electricity tariffs.

Units convention:
- Power: kilowatts [kW]
- Energy: kilowatt-hours [kWh]
- Time: minutes [minutes] and discrete 15-minute time steps [unitless]
- Price: currency per kilowatt-hour [$/kWh]

Distinction:
- [SIMULATED]: Ground-truth physical variables inside the simulator.
- [MEASURED]: Telemetry observed across the hardware abstraction layer (HAL).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class EVSession:
    """Represents a single electric vehicle charging session.

    Parameters
    ----------
    arrival_min : int
        [MEASURED & SIMULATED] Arrival / plug-in time in elapsed minutes
        from the simulation start [minutes].
    departure_min : int
        [MEASURED & SIMULATED] Departure / unplug time in elapsed minutes
        from the simulation start [minutes]. Must be strictly greater than arrival_min.
    energy_req_kwh : float
        [MEASURED & SIMULATED] Total electrical energy requested by the EV
        to reach target state of charge, in kilowatt-hours [kWh].
    max_rate_kw : float
        [MEASURED & SIMULATED] Maximum continuous active charging power
        accepted by the EV onboard charger / EVSE, in kilowatts [kW].
    session_id : str, default ""
        Unique identifier for the vehicle session (e.g. 'EV_01').
    delivered_energy_kwh : float, default 0.0
        [SIMULATED] Ground-truth cumulative energy transferred into the
        battery during this session, in kilowatt-hours [kWh].
    """

    arrival_min: int
    departure_min: int
    energy_req_kwh: float
    max_rate_kw: float
    session_id: str = ""
    delivered_energy_kwh: float = 0.0

    def __post_init__(self) -> None:
        if self.departure_min <= self.arrival_min:
            raise ValueError(
                f"departure_min ({self.departure_min}) must be strictly greater "
                f"than arrival_min ({self.arrival_min})."
            )
        if self.energy_req_kwh < 0.0:
            raise ValueError(f"energy_req_kwh ({self.energy_req_kwh}) cannot be negative.")
        if self.max_rate_kw <= 0.0:
            raise ValueError(f"max_rate_kw ({self.max_rate_kw}) must be positive.")

    @property
    def duration_min(self) -> int:
        """Total dwell time of the vehicle at the charger in minutes [minutes]."""
        return self.departure_min - self.arrival_min

    @property
    def remaining_energy_req_kwh(self) -> float:
        """[SIMULATED] Remaining unfulfilled energy needed to reach target [kWh]."""
        return max(0.0, self.energy_req_kwh - self.delivered_energy_kwh)

    def is_connected(self, t_min: int) -> bool:
        """Check if vehicle is physically plugged in at elapsed minute t_min."""
        return self.arrival_min <= t_min < self.departure_min


@dataclass(frozen=True)
class Feeder:
    """Distribution substation transformer and feeder capacity specifications.

    Parameters
    ----------
    transformer_limit_kw : float
        [SIMULATED & MEASURED] Continuous active power rating limit of the
        substation transformer in kilowatts [kW]. Operating above this threshold
        causes thermal degradation and insulation breakdown.
    base_load_profile_kw : List[float]
        [SIMULATED] Background non-EV grid active power demand per 15-minute
        time step in kilowatts [kW].
    """

    transformer_limit_kw: float
    base_load_profile_kw: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.transformer_limit_kw <= 0.0:
            raise ValueError("transformer_limit_kw must be positive.")


@dataclass(frozen=True)
class Tariff:
    """Time-of-Use (TOU) electricity pricing schedule.

    Parameters
    ----------
    price_per_kwh_per_15min : List[float]
        Energy electricity price per 15-minute time step in dollars per
        kilowatt-hour [$/kWh].
    """

    price_per_kwh_per_15min: List[float] = field(default_factory=list)
