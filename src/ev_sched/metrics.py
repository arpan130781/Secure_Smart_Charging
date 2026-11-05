"""Evaluation metrics for grid security, charging cost, and energy quality of service.

Standard Units:
- Overload duration: minutes [minutes]
- Electricity cost: dollars [$]
- Unmet energy: kilowatt-hours [kWh]
- Peak feeder active power: kilowatts [kW]
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ev_sched.models import EVSession, Tariff


@dataclass(frozen=True)
class RunMetrics:
    """Consolidated performance metrics for an EV charging run.

    Parameters
    ----------
    overload_minutes : float
        Total duration that feeder active power exceeded transformer limit in minutes [minutes].
    total_cost_usd : float
        Total electricity procurement cost in dollars [$].
    unmet_energy_kwh : float
        Sum of unfulfilled energy demands across all EV sessions in kilowatt-hours [kWh].
    peak_load_kw : float
        Maximum total active power demand observed on the substation feeder in kilowatts [kW].
    """

    overload_minutes: float
    total_cost_usd: float
    unmet_energy_kwh: float
    peak_load_kw: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to a dictionary of rounded floats."""
        return {
            "overload_minutes": round(self.overload_minutes, 1),
            "total_cost_usd": round(self.total_cost_usd, 2),
            "unmet_energy_kwh": round(self.unmet_energy_kwh, 2),
            "peak_load_kw": round(self.peak_load_kw, 2),
        }


def compute_overload_minutes(
    feeder_load_kw: Sequence[float],
    limit_kw: float,
    time_step_min: int = 15,
) -> float:
    """Compute total minutes the feeder operated above its transformer limit.

    Parameters
    ----------
    feeder_load_kw : Sequence[float]
        Time-series of total feeder active power [kW].
    limit_kw : float
        Transformer continuous active power rating [kW].
    time_step_min : int, default 15
        Duration of each interval in minutes [minutes].

    Returns
    -------
    float
        Overload duration in minutes [minutes].
    """
    overload_steps = sum(1 for p in feeder_load_kw if p > limit_kw + 1e-4)
    return float(overload_steps * time_step_min)


def compute_electricity_cost_usd(
    ev_power_kw: Sequence[float],
    tariff: Tariff,
    time_step_min: int = 15,
) -> float:
    """Calculate total electricity cost incurred by EV charging in dollars [$].

    Parameters
    ----------
    ev_power_kw : Sequence[float]
        Time-series of total EV charging active power [kW].
    tariff : Tariff
        Time-of-Use price schedule [$/kWh].
    time_step_min : int, default 15
        Step duration in minutes [minutes].

    Returns
    -------
    float
        Total energy cost in dollars [$].
    """
    dt_hours = time_step_min / 60.0
    prices = list(tariff.price_per_kwh_per_15min)
    num_steps = len(ev_power_kw)
    if len(prices) < num_steps:
        pad = prices[-1] if prices else 0.15
        prices.extend([pad] * (num_steps - len(prices)))

    cost = sum(p * kw * dt_hours for p, kw in zip(prices[:num_steps], ev_power_kw))
    return float(cost)


def compute_unmet_energy_kwh(sessions: Sequence[EVSession]) -> float:
    """Sum total unfulfilled energy demands across all EV sessions in kilowatt-hours [kWh].

    Parameters
    ----------
    sessions : Sequence[EVSession]
        List of EV charging sessions with final delivered energies.

    Returns
    -------
    float
        Total unfulfilled energy in kilowatt-hours [kWh].
    """
    return float(sum(max(0.0, s.energy_req_kwh - s.delivered_energy_kwh) for s in sessions))


def compute_peak_load_kw(feeder_load_kw: Sequence[float]) -> float:
    """Find maximum active power flow reached on the feeder in kilowatts [kW].

    Parameters
    ----------
    feeder_load_kw : Sequence[float]
        Time-series of total feeder active power [kW].

    Returns
    -------
    float
        Peak active power in kilowatts [kW].
    """
    return float(max(feeder_load_kw)) if feeder_load_kw else 0.0


def compute_run_metrics(
    feeder_load_kw: Sequence[float],
    ev_power_kw: Sequence[float],
    sessions: Sequence[EVSession],
    tariff: Tariff,
    transformer_limit_kw: float,
    time_step_min: int = 15,
) -> RunMetrics:
    """Evaluate consolidated grid, cost, and energy QoS metrics for a simulation run."""
    return RunMetrics(
        overload_minutes=compute_overload_minutes(feeder_load_kw, transformer_limit_kw, time_step_min),
        total_cost_usd=compute_electricity_cost_usd(ev_power_kw, tariff, time_step_min),
        unmet_energy_kwh=compute_unmet_energy_kwh(sessions),
        peak_load_kw=compute_peak_load_kw(feeder_load_kw),
    )
