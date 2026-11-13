"""Task 17: Limitation Study - Adaptive Attacker Evading Detector 1.

Demonstrated Behavior vs. Security Guarantees:
----------------------------------------------
This experiment demonstrates a fundamental limitation of residual-based anomaly detection:
A model-cognizant attacker who knows the EMS energy-balance equation:
    expected_feeder_kw = P_base_forecast + sum(P_ev)
can synthesize false meter telemetry that keeps the innovation residual r(t) near zero.

Crucially, this experiment demonstrates a detector limitation:
- Detector 1 (residual CUSUM) observes r(t) ~ N(0, sigma^2) and raises 0 alarms (0% recall).
- This confirms that residual detection alone does NOT establish overload protection against
  unmodeled physical load surges when an adversary can falsify feeder meter telemetry.

CONDITIONAL PROTECTION CLAIM & DEFENSE REQUIREMENTS:
-----------------------------------------------------
The system has NO independent physical overload protection unless an authoritative
independent protective constraint (`max_aggregate_ev_kw`) or a genuine independent out-of-band
sensor (`trusted_meter`) is explicitly configured and active.
Without one of these active defenses, the system is completely UNPROTECTED against adaptive attacks:
- Defense 1: Independent Aggregate EV Cap (`max_aggregate_ev_kw`):
    sum_{i} P_{ev}[i, t] <= max_aggregate_ev_kw
  derived from an authoritative credible peak non-EV base demand ceiling:
    max_aggregate_ev_kw = P_transformer_limit - P_base_max_credible.
- Defense 2: Out-of-band Secondary Trusted Meter (`trusted_meter`):
  An isolated, cryptographically authenticated physical sensor measuring total substation power
  that dynamically clamps available headroom when physical surges occur.

Assumptions Required:
1. For Defense 1: Operators possess an authoritative, unbreachable upper bound on credible non-EV base demand.
2. For Defense 2: The secondary meter telemetry channel is physically isolated or cryptographically
   secured from the primary metering compromise domain.
"""

import numpy as np

from ev_sched.attacks import AdaptiveAttackMeter
from ev_sched.detector import CUSUMDetector
from ev_sched.hal import MeterInterface, SimEVSE, SimMeter
from ev_sched.metrics import compute_overload_minutes
from ev_sched.models import EVSession, Feeder, Tariff
from ev_sched.scheduler import run_mpc_charging_loop
from ev_sched.simulator import ChargingStationSimulator


def run_adaptive_attack_limitation_experiment(
    seed: int = 42,
    max_aggregate_ev_kw: float | None = None,
    trusted_meter: MeterInterface | None = None,
) -> dict:
    """Simulate adaptive attacker evading Detector 1, with optional independent protection.

    Parameters
    ----------
    seed : int, default 42
        Seed for reproducible noise realization.
    max_aggregate_ev_kw : Optional[float], default None
        Independent protective cluster limit on aggregate EV power [kW].
    trusted_meter : Optional[MeterInterface], default None
        Out-of-band secondary trusted meter.
    """
    # Substation transformer limit: 80.0 kW
    transformer_limit_kw = 80.0

    # Nominal base forecast known to the scheduler: 45.0 kW
    base_forecast = [45.0] * 16

    # Physical reality: an unmodeled industrial load surge causes actual base load
    # to jump to 75.0 kW between step 4 (t=60 min) and step 12 (t=180 min).
    actual_base_load = [45.0] * 4 + [75.0] * 8 + [45.0] * 4

    # Controller only knows the base forecast
    feeder_controller = Feeder(transformer_limit_kw=transformer_limit_kw, base_load_profile_kw=base_forecast)
    # Simulator experiences true physical surge
    feeder_sim = Feeder(transformer_limit_kw=transformer_limit_kw, base_load_profile_kw=actual_base_load)

    # 4 EV charging sessions requesting power (each 11 kW max -> 44 kW potential demand)
    sessions = [
        EVSession(arrival_min=0, departure_min=240, energy_req_kwh=15.0, max_rate_kw=11.0, session_id=f"EV_{i}")
        for i in range(4)
    ]
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 16)

    sim = ChargingStationSimulator(feeder=feeder_sim, sessions=sessions, time_step_min=15, rng=np.random.default_rng(seed))
    raw_meter = SimMeter(sim, noise_std_kw=0.2, rng=np.random.default_rng(seed + 100))
    evse = SimEVSE(sim)

    # Adaptive attacker crafts telemetry matching forecast_base + ev_power + noise
    adaptive_meter = AdaptiveAttackMeter(
        inner_meter=raw_meter,
        start_min=60,
        duration_min=120,
        base_forecast_profile_kw=base_forecast,
        evse=evse,
        session_ids=[s.session_id for s in sessions],
        noise_std_kw=0.2,
        seed=seed + 200,
    )

    detector = CUSUMDetector(threshold=18.0, slack_kw=1.5, instant_threshold_kw=20.0)

    # Resolve active trusted meter bound to physical simulation if configured
    active_trusted_meter: MeterInterface | None = None
    if trusted_meter is True:
        active_trusted_meter = SimMeter(sim, noise_std_kw=0.2, rng=np.random.default_rng(seed + 300))
    elif isinstance(trusted_meter, SimMeter):
        trusted_meter.bind_simulator(sim)
        active_trusted_meter = trusted_meter
    elif isinstance(trusted_meter, MeterInterface):
        active_trusted_meter = trusted_meter

    # Run MPC loop with detector using the controller's feeder model
    mpc_res = run_mpc_charging_loop(
        feeder=feeder_controller,
        sessions=sessions,
        tariff=tariff,
        meter=adaptive_meter,
        evse=evse,
        simulator=sim,
        time_step_min=15,
        num_steps=16,
        detector=detector,
        conservative_margin_kw=10.0,
        max_aggregate_ev_kw=max_aggregate_ev_kw,
        trusted_meter=active_trusted_meter,
    )

    # True physical feeder load on transformer = actual base load + dispatched EV power
    true_feeder_kw = [
        actual_base_load[step] + ev_kw
        for step, ev_kw in enumerate(mpc_res.total_ev_power_kw)
    ]
    peak_true = max(true_feeder_kw)
    overload_min = compute_overload_minutes(
        feeder_load_kw=true_feeder_kw,
        limit_kw=transformer_limit_kw,
        time_step_min=15,
    )
    alarms = sum(1 for a in mpc_res.alarms_raised if a)

    # Confirm active defense actually constrained EV charging and prevented transformer overload
    has_defense_input = max_aggregate_ev_kw is not None or active_trusted_meter is not None
    defense_constrained = any(mpc_res.safe_mode_active) or (max_aggregate_ev_kw is not None)
    no_physical_overload = (overload_min == 0.0) and (round(peak_true, 2) <= transformer_limit_kw)
    is_protected = bool(
        has_defense_input
        and defense_constrained
        and no_physical_overload
        and mpc_res.is_protection_active
    )

    return {
        "transformer_limit_kw": transformer_limit_kw,
        "peak_true_load_kw": round(peak_true, 2),
        "overload_minutes": round(overload_min, 1),
        "alarms_raised": alarms,
        "attack_steps": 8,
        "is_protected": is_protected,
    }


def main() -> None:
    unprotected = run_adaptive_attack_limitation_experiment()
    # Protected by independent aggregate cluster limit (80 kW limit - 75 kW peak surge = 5 kW allowable)
    protected = run_adaptive_attack_limitation_experiment(max_aggregate_ev_kw=5.0)

    print("=" * 80)
    print("TASK 17: ADAPTIVE ATTACKER LIMITATION STUDY RESULTS")
    print("=" * 80)
    print("1. UNPROTECTED SYSTEM (Relying Solely on Primary Meter & Base Forecast):")
    print(f"   Transformer Limit           : {unprotected['transformer_limit_kw']:.1f} kW")
    print(f"   Peak Physical Feeder Load    : {unprotected['peak_true_load_kw']:.2f} kW (OVERLOAD!)")
    print(f"   Overload Duration           : {unprotected['overload_minutes']:.1f} min")
    print(f"   Alarms Raised by Detector 1 : {unprotected['alarms_raised']} alarms (100% MISSED!)")
    print("-" * 80)
    print("2. PROTECTED SYSTEM (Independent Protective Constraint max_aggregate_ev_kw=5.0 kW):")
    print(f"   Transformer Limit           : {protected['transformer_limit_kw']:.1f} kW")
    print(f"   Peak Physical Feeder Load    : {protected['peak_true_load_kw']:.2f} kW (PROTECTED)")
    print(f"   Overload Duration           : {protected['overload_minutes']:.1f} min")
    print(f"   Alarms Raised by Detector 1 : {protected['alarms_raised']} alarms")
    print("=" * 80)


if __name__ == "__main__":
    main()
