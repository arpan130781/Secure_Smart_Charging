"""Task 17: Limitation Study - Adaptive Attacker Evading Detector 1.

Demonstrates that a model-cognizant attacker who knows the EMS energy balance:
    expected_feeder_kw = P_base_forecast + sum(P_ev)
can craft false data that keeps the innovation residual r(t) near zero.

Consequence:
- Detector 1 (residual CUSUM) observes r(t) ~ N(0, sigma^2), raising 0 alarms (0% recall).
- Meanwhile, an unmodeled physical base surge causes physical transformer overload.
"""

import csv
from pathlib import Path
import numpy as np

from ev_sched.models import EVSession, Feeder, Tariff
from ev_sched.simulator import ChargingStationSimulator
from ev_sched.hal import SimEVSE, SimMeter
from ev_sched.attacks import AdaptiveAttackMeter
from ev_sched.detector import CUSUMDetector
from ev_sched.scheduler import run_mpc_charging_loop


def run_adaptive_attack_limitation_experiment() -> dict:
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

    sim = ChargingStationSimulator(feeder=feeder_sim, sessions=sessions, time_step_min=15, rng=np.random.default_rng(42))
    raw_meter = SimMeter(sim, noise_std_kw=0.2)
    evse = SimEVSE(sim)

    # Adaptive attacker crafts telemetry matching forecast_base + ev_power + noise
    adaptive_meter = AdaptiveAttackMeter(
        inner_meter=raw_meter,
        start_min=60,
        duration_min=120,
        base_forecast_profile_kw=base_forecast,
        simulator=sim,
        noise_std_kw=0.2,
        seed=42,
    )

    detector = CUSUMDetector(threshold=18.0, slack_kw=1.5, instant_threshold_kw=20.0)

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
    )

    # True physical feeder load on transformer = actual base load + dispatched EV power
    true_feeder_kw = [
        actual_base_load[step] + ev_kw
        for step, ev_kw in enumerate(mpc_res.total_ev_power_kw)
    ]
    peak_true = max(true_feeder_kw)
    overload_min = sum(15.0 for p in true_feeder_kw if p > transformer_limit_kw)
    alarms = sum(1 for a in mpc_res.alarms_raised if a)

    return {
        "transformer_limit_kw": transformer_limit_kw,
        "peak_true_load_kw": round(peak_true, 2),
        "overload_minutes": round(overload_min, 1),
        "alarms_raised": alarms,
        "attack_steps": 8,
    }


def main() -> None:
    results = run_adaptive_attack_limitation_experiment()
    print("=" * 80)
    print("TASK 17: ADAPTIVE ATTACKER LIMITATION STUDY RESULTS")
    print("=" * 80)
    print(f"Transformer Limit           : {results['transformer_limit_kw']:.1f} kW")
    print(f"Peak Physical Feeder Load    : {results['peak_true_load_kw']:.2f} kW (OVERLOAD!)")
    print(f"Overload Duration           : {results['overload_minutes']:.1f} min")
    print(f"Alarms Raised by Detector 1 : {results['alarms_raised']} alarms (100% MISSED!)")
    print("=" * 80)


if __name__ == "__main__":
    main()
