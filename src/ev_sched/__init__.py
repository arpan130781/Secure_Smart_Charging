"""ev_sched: Secure Smart EV Charging package."""

from ev_sched.models import EVSession, Feeder, Tariff
from ev_sched.simulator import (
    ChargingStationSimulator,
    SimulationResult,
    generate_synthetic_sessions,
    run_uncontrolled_charging,
)
from ev_sched.hal import EVSEInterface, MeterInterface, SimEVSE, SimMeter
from ev_sched.scheduler import (
    OptimizationResult,
    MPCSimulationResult,
    solve_charging_schedule,
    run_mpc_charging_loop,
)
from ev_sched.attacks import (
    ConstantBiasMeter,
    RampAttackMeter,
    ReplayAttackMeter,
    create_attacked_meter,
)
from ev_sched.detector import (
    DetectorAlert,
    CUSUMDetector,
    IsolationForestDetector,
)
from ev_sched.metrics import (
    RunMetrics,
    compute_overload_minutes,
    compute_electricity_cost_usd,
    compute_unmet_energy_kwh,
    compute_peak_load_kw,
    compute_run_metrics,
)

__all__ = [
    "EVSession",
    "Feeder",
    "Tariff",
    "SimulationResult",
    "ChargingStationSimulator",
    "generate_synthetic_sessions",
    "run_uncontrolled_charging",
    "MeterInterface",
    "EVSEInterface",
    "SimMeter",
    "SimEVSE",
    "OptimizationResult",
    "MPCSimulationResult",
    "solve_charging_schedule",
    "run_mpc_charging_loop",
    "ConstantBiasMeter",
    "RampAttackMeter",
    "ReplayAttackMeter",
    "create_attacked_meter",
    "DetectorAlert",
    "CUSUMDetector",
    "IsolationForestDetector",
    "RunMetrics",
    "compute_overload_minutes",
    "compute_electricity_cost_usd",
    "compute_unmet_energy_kwh",
    "compute_peak_load_kw",
    "compute_run_metrics",
]
