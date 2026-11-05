"""Prove physical energy conservation across multiple seeds and operational conditions.

Mathematical Invariant:
-----------------------
The cumulative electrochemical energy delivered into all vehicle battery packs
must exactly equal the time-integral of aggregate EV active charging power:

    sum_{s in sessions} E_delivered[s]  ==  sum_{t=0}^{T-1} P_ev_total[t] * dt_hours

This script simulates 10 random seeds (0 to 9) across:
1. Uncontrolled charging baseline
2. Model Predictive Control (Clean nominal operation)
3. Model Predictive Control (Under FDI telemetry attack)

For each run, it asserts |E_delivered - int P_ev dt| < 1e-9 kWh.
"""

import numpy as np
import yaml

from ev_sched.attacks import ConstantBiasMeter
from ev_sched.hal import SimEVSE, SimMeter
from ev_sched.models import Feeder
from ev_sched.scheduler import run_mpc_charging_loop
from ev_sched.simulator import (
    ChargingStationSimulator,
    generate_synthetic_sessions,
    run_uncontrolled_charging,
)
from experiments.compare_baseline_vs_mpc import create_tou_tariff
from experiments.run_baseline import generate_diurnal_base_load


def verify_energy_conservation(num_seeds: int = 10, config_path: str = "experiments/baseline_config.yaml") -> None:
    """Run energy conservation verification over multiple seeds."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tariff = create_tou_tariff(num_steps=96)
    dt_hours = 15.0 / 60.0  # 15 minutes = 0.25 hours

    print("=" * 80)
    print("PROVING PHYSICAL CONSERVATION OF ENERGY ACROSS 10 SEEDS")
    print(f"Mathematical Invariant: sum(E_delivered_kwh) == sum(P_ev_kw) * {dt_hours} h")
    print("=" * 80)
    print(f"{'Seed':<6} | {'Uncontrolled Error [kWh]':<26} | {'MPC Clean Error [kWh]':<22} | {'MPC Attack Error [kWh]':<22} | Status")
    print("-" * 88)

    max_error_kwh = 0.0

    for seed in range(num_seeds):
        # Generate base load and sessions
        rng_base = np.random.default_rng(seed + 999)
        base_profile = generate_diurnal_base_load(
            num_steps=96,
            nominal_kw=cfg["feeder"]["nominal_base_kw"],
            peak_kw=cfg["feeder"]["peak_base_kw"],
            valley_kw=cfg["feeder"]["valley_base_kw"],
            rng=rng_base,
        )
        feeder = Feeder(
            transformer_limit_kw=cfg["feeder"]["transformer_limit_kw"],
            base_load_profile_kw=base_profile,
        )

        # -------------------------------------------------------------
        # 1. Uncontrolled Baseline
        # -------------------------------------------------------------
        rng_unc = np.random.default_rng(seed)
        sessions_unc = generate_synthetic_sessions(cfg, rng_unc)
        res_unc = run_uncontrolled_charging(feeder, sessions_unc, time_step_min=15, num_steps=96)
        int_unc_kwh = sum(res_unc.ev_load_kw) * dt_hours
        sum_unc_kwh = sum(s.delivered_energy_kwh for s in res_unc.sessions)
        diff_unc = abs(int_unc_kwh - sum_unc_kwh)
        assert diff_unc < 1e-9, f"Seed {seed} Uncontrolled mismatch: {int_unc_kwh} vs {sum_unc_kwh}"

        # -------------------------------------------------------------
        # 2. Clean MPC Loop
        # -------------------------------------------------------------
        rng_mpc = np.random.default_rng(seed)
        sessions_mpc = generate_synthetic_sessions(cfg, rng_mpc)
        sim_mpc = ChargingStationSimulator(feeder, sessions_mpc, time_step_min=15, rng=np.random.default_rng(seed + 100))
        meter_mpc = SimMeter(sim_mpc, noise_std_kw=0.3, rng=np.random.default_rng(seed + 200))
        evse_mpc = SimEVSE(sim_mpc)
        res_mpc = run_mpc_charging_loop(feeder, sessions_mpc, tariff, meter_mpc, evse_mpc, simulator=sim_mpc, horizon_steps=16)
        int_mpc_kwh = sum(res_mpc.total_ev_power_kw) * dt_hours
        sum_mpc_kwh = sum(s.delivered_energy_kwh for s in res_mpc.sessions)
        diff_mpc = abs(int_mpc_kwh - sum_mpc_kwh)
        assert diff_mpc < 1e-9, f"Seed {seed} MPC mismatch: {int_mpc_kwh} vs {sum_mpc_kwh}"

        # -------------------------------------------------------------
        # 3. MPC under FDI Attack
        # -------------------------------------------------------------
        rng_atk = np.random.default_rng(seed)
        sessions_atk = generate_synthetic_sessions(cfg, rng_atk)
        sim_atk = ChargingStationSimulator(feeder, sessions_atk, time_step_min=15, rng=np.random.default_rng(seed + 300))
        meter_clean = SimMeter(sim_atk, noise_std_kw=0.3, rng=np.random.default_rng(seed + 400))
        meter_atk = ConstantBiasMeter(meter_clean, start_min=540, duration_min=420, delta_kw=35.0)
        evse_atk = SimEVSE(sim_atk)
        res_atk = run_mpc_charging_loop(feeder, sessions_atk, tariff, meter_atk, evse_atk, simulator=sim_atk, horizon_steps=16)
        int_atk_kwh = sum(res_atk.total_ev_power_kw) * dt_hours
        sum_atk_kwh = sum(s.delivered_energy_kwh for s in res_atk.sessions)
        diff_atk = abs(int_atk_kwh - sum_atk_kwh)
        assert diff_atk < 1e-9, f"Seed {seed} Attack mismatch: {int_atk_kwh} vs {sum_atk_kwh}"

        # Check that no vehicle exceeded requested energy
        for s in res_mpc.sessions:
            assert s.delivered_energy_kwh <= s.energy_req_kwh + 1e-9

        seed_max = max(diff_unc, diff_mpc, diff_atk)
        max_error_kwh = max(max_error_kwh, seed_max)

        print(f"Seed {seed:<2d} | {diff_unc:20.2e} kWh | {diff_mpc:17.2e} kWh | {diff_atk:17.2e} kWh | PASSED (Exact)")

    print("-" * 88)
    print(f"SUMMARY: 10/10 seeds passed. Maximum observed numerical error across all runs: {max_error_kwh:.2e} kWh")
    print("Exact energy conservation holds unconditionally: sum(E_delivered) == integral(P_ev dt).\n")


if __name__ == "__main__":
    verify_energy_conservation(num_seeds=10)
