"""ev_sched: Secure Smart EV Charging package."""

from ev_sched.attacks import (
    AdaptiveAttackMeter,
    ConstantBiasMeter,
    RampAttackMeter,
    ReplayAttackMeter,
    SimilarLoadReplayAttackMeter,
    create_attacked_meter,
)
from ev_sched.detector import (
    CUSUMDetector,
    DetectorAlert,
    IsolationForestDetector,
    KalmanChiSquareDetector,
)
from ev_sched.hal import EVSEInterface, MeterInterface, SimEVSE, SimMeter
from ev_sched.metrics import (
    RunMetrics,
    compute_electricity_cost_usd,
    compute_overload_minutes,
    compute_peak_load_kw,
    compute_run_metrics,
    compute_unmet_energy_kwh,
)
from ev_sched.models import EVSession, Feeder, Tariff, resolve_session_ids
from ev_sched.scheduler import (
    MPCSimulationResult,
    OptimizationResult,
    run_mpc_charging_loop,
    solve_charging_schedule,
    solve_water_filling_step,
    validate_schedule_constraints,
)
from ev_sched.simulator import (
    ChargingStationSimulator,
    SimulationResult,
    generate_base_forecast_with_ar1_error,
    generate_synthetic_sessions,
    run_uncontrolled_charging,
)

__all__ = [
    "AdaptiveAttackMeter",
    "CUSUMDetector",
    "ChargingStationSimulator",
    "ConstantBiasMeter",
    "DetectorAlert",
    "EVSEInterface",
    "EVSession",
    "Feeder",
    "IsolationForestDetector",
    "KalmanChiSquareDetector",
    "MPCSimulationResult",
    "MeterInterface",
    "OptimizationResult",
    "RampAttackMeter",
    "ReplayAttackMeter",
    "RunMetrics",
    "SimEVSE",
    "SimMeter",
    "SimilarLoadReplayAttackMeter",
    "SimulationResult",
    "Tariff",
    "compute_electricity_cost_usd",
    "compute_overload_minutes",
    "compute_peak_load_kw",
    "compute_run_metrics",
    "compute_unmet_energy_kwh",
    "create_attacked_meter",
    "generate_base_forecast_with_ar1_error",
    "generate_synthetic_sessions",
    "resolve_session_ids",
    "run_mpc_charging_loop",
    "run_uncontrolled_charging",
    "solve_charging_schedule",
    "solve_water_filling_step",
    "validate_schedule_constraints",
]
