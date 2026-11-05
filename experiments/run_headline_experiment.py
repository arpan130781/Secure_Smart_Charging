"""Run Task 16: Generate headline figure and summary table.

Compares 3 conditions across 24 hours (96 steps at 15-min intervals):
1. Normal: Clean MPC scheduling without cyber-attack.
2. Under Attack: Constant bias FDI (-35 kW) without detector (unmitigated).
3. Under Attack with Detection: Constant bias FDI (-35 kW) with CUSUM detector
   triggering safe mode (restoring expected base load and applying conservative margin).

Outputs:
- results/headline.png: 24-hour feeder load profiles vs transformer limit.
- results/headline.csv: Comparison of peak load, overload minutes, cost, and unmet energy.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from ev_sched.attacks import ConstantBiasMeter
from ev_sched.detector import CUSUMDetector
from ev_sched.hal import SimEVSE, SimMeter
from ev_sched.metrics import (
    compute_electricity_cost_usd,
    compute_overload_minutes,
    compute_peak_load_kw,
    compute_unmet_energy_kwh,
)
from ev_sched.models import EVSession, Feeder
from ev_sched.scheduler import run_mpc_charging_loop
from ev_sched.simulator import (
    ChargingStationSimulator,
    generate_synthetic_sessions,
)
from experiments.compare_baseline_vs_mpc import create_tou_tariff
from experiments.run_baseline import generate_diurnal_base_load


def run_headline_simulation(
    config_path: Path,
    seed: int = 42,
    num_steps: int = 96,
    time_step_min: int = 15,
) -> tuple[dict[str, list[float]], dict[str, dict[str, float]]]:
    """Execute the three headline conditions for a reproducible 24-hour run."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    feeder_limit_kw = float(config["feeder"]["transformer_limit_kw"])

    # 1. Generate base load profile and tariff
    rng_base = np.random.default_rng(seed)
    base_load = generate_diurnal_base_load(
        num_steps=num_steps,
        nominal_kw=config["feeder"]["nominal_base_kw"],
        peak_kw=config["feeder"]["peak_base_kw"],
        valley_kw=config["feeder"]["valley_base_kw"],
        rng=rng_base,
    )
    tariff = create_tou_tariff(num_steps=num_steps)

    # 2. Generate EV charging sessions
    rng_sessions = np.random.default_rng(seed + 100)
    sessions_template = generate_synthetic_sessions(config, rng_sessions)

    # Attack configuration: -35 kW from 09:00 (540 min) to 16:00 (960 min)
    start_min = 540
    duration_min = 420
    delta_kw = 35.0

    conditions = ["normal", "under_attack", "under_attack_with_detection"]
    time_series_data: dict[str, list[float]] = {}
    summary_metrics: dict[str, dict[str, float]] = {}

    for cond in conditions:
        # Clone fresh sessions with 0 delivered energy
        curr_sessions = [
            EVSession(
                arrival_min=s.arrival_min,
                departure_min=s.departure_min,
                energy_req_kwh=s.energy_req_kwh,
                max_rate_kw=s.max_rate_kw,
                session_id=s.session_id,
            )
            for s in sessions_template
        ]

        feeder = Feeder(transformer_limit_kw=feeder_limit_kw, base_load_profile_kw=list(base_load))
        rng_sim = np.random.default_rng(seed + 200)
        sim = ChargingStationSimulator(feeder=feeder, sessions=curr_sessions, time_step_min=time_step_min, rng=rng_sim)
        raw_meter = SimMeter(sim, noise_std_kw=0.3)
        evse = SimEVSE(sim)

        if cond == "normal":
            meter = raw_meter
            detector = None
        elif cond == "under_attack":
            meter = ConstantBiasMeter(
                inner_meter=raw_meter,
                start_min=start_min,
                duration_min=duration_min,
                delta_kw=delta_kw,
            )
            detector = None
        elif cond == "under_attack_with_detection":
            meter = ConstantBiasMeter(
                inner_meter=raw_meter,
                start_min=start_min,
                duration_min=duration_min,
                delta_kw=delta_kw,
            )
            detector = CUSUMDetector(threshold=18.0, slack_kw=1.5, instant_threshold_kw=20.0)

        # Execute MPC loop
        mpc_res = run_mpc_charging_loop(
            feeder=feeder,
            sessions=curr_sessions,
            tariff=tariff,
            meter=meter,
            evse=evse,
            simulator=sim,
            time_step_min=time_step_min,
            num_steps=num_steps,
            horizon_steps=16,
            detector=detector,
            conservative_margin_kw=10.0,
            safe_mode_cooldown_steps=4,
        )

        time_series_data[cond] = mpc_res.true_feeder_kw

        # Compute summary metrics
        peak_load = compute_peak_load_kw(mpc_res.true_feeder_kw)
        overload_min = compute_overload_minutes(
            mpc_res.true_feeder_kw, feeder_limit_kw, time_step_min=time_step_min
        )
        cost_usd = compute_electricity_cost_usd(
            mpc_res.total_ev_power_kw, tariff, time_step_min=time_step_min
        )
        unmet_kwh = compute_unmet_energy_kwh(curr_sessions)

        summary_metrics[cond] = {
            "peak_true_load_kw": round(peak_load, 2),
            "overload_minutes": round(overload_min, 1),
            "total_cost_usd": round(cost_usd, 2),
            "unmet_energy_kwh": round(unmet_kwh, 2),
        }

    return time_series_data, summary_metrics


def plot_headline_figure(
    time_series_data: dict[str, list[float]],
    transformer_limit_kw: float,
    output_png_path: Path,
    time_step_min: int = 15,
) -> None:
    """Generate 3-curve / 3-panel publication figure in plain matplotlib style."""
    num_steps = len(time_series_data["normal"])
    hours = np.array([step * time_step_min / 60.0 for step in range(num_steps)])

    _fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)

    # Panel 1: Normal
    ax0 = axes[0]
    ax0.plot(hours, time_series_data["normal"], color="#1f77b4", linewidth=1.8, label="True Feeder Load [kW]")
    ax0.axhline(transformer_limit_kw, color="black", linestyle="--", linewidth=1.2, label=f"Transformer Limit ({transformer_limit_kw:.0f} kW)")
    ax0.set_title("(a) Normal Operation (Clean MPC)", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Power [kW]", fontsize=10)
    ax0.grid(True, linestyle=":", alpha=0.6)
    ax0.legend(loc="upper left", framealpha=0.9)

    # Panel 2: Under Attack (Unmitigated)
    ax1 = axes[1]
    ax1.axvspan(9.0, 16.0, color="#ffcccc", alpha=0.4, label="FDI Attack Window (-35 kW)")
    ax1.plot(hours, time_series_data["under_attack"], color="#d62728", linewidth=1.8, label="True Feeder Load [kW] (Corrupted)")
    ax1.axhline(transformer_limit_kw, color="black", linestyle="--", linewidth=1.2, label=f"Transformer Limit ({transformer_limit_kw:.0f} kW)")
    # Highlight overload area above 120 kW
    y_over = np.maximum(transformer_limit_kw, np.array(time_series_data["under_attack"]))
    ax1.fill_between(hours, transformer_limit_kw, y_over, color="#d62728", alpha=0.3, label="Transformer Overload [kW]")
    ax1.set_title("(b) Under Attack (Unmitigated FDI: -35 kW under-reporting)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Power [kW]", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.9)

    # Panel 3: Under Attack with Detection (Mitigated)
    ax2 = axes[2]
    ax2.axvspan(9.0, 16.0, color="#e6f2ff", alpha=0.4, label="FDI Attack Window (-35 kW)")
    ax2.plot(hours, time_series_data["under_attack_with_detection"], color="#2ca02c", linewidth=1.8, label="True Feeder Load [kW] (Mitigated)")
    ax2.axhline(transformer_limit_kw, color="black", linestyle="--", linewidth=1.2, label=f"Transformer Limit ({transformer_limit_kw:.0f} kW)")
    ax2.axhline(transformer_limit_kw - 10.0, color="#2ca02c", linestyle=":", linewidth=1.2, label="Safe Mode Headroom (110 kW)")
    ax2.set_title("(c) Under Attack with Detection & Safe Mode (Mitigated)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time of Day [Hours]", fontsize=10)
    ax2.set_ylabel("Power [kW]", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.xlim(0.0, 24.0)
    plt.xticks(np.arange(0, 25, 2))
    plt.tight_layout()

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png_path, dpi=300)
    plt.close()
    print(f"Saved headline figure to: {output_png_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "experiments" / "baseline_config.yaml"
    results_dir = project_root / "results"

    print("Running Task 16: Headline 24-hour comparative simulation...")
    time_series, metrics = run_headline_simulation(config_path=config_path, seed=42)

    # Save summary table to results/headline.csv
    csv_path = results_dir / "headline.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "peak_true_load_kw", "overload_minutes", "total_cost_usd", "unmet_energy_kwh"])
        for cond, m in metrics.items():
            writer.writerow([
                cond,
                m["peak_true_load_kw"],
                m["overload_minutes"],
                m["total_cost_usd"],
                m["unmet_energy_kwh"],
            ])
    print(f"Saved headline metrics table to: {csv_path}")

    # Plot results/headline.png
    png_path = results_dir / "headline.png"
    plot_headline_figure(time_series, transformer_limit_kw=120.0, output_png_path=png_path)

    print("\n" + "=" * 80)
    print("TASK 16: HEADLINE EVALUATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Condition':<30} {'Peak True Load':<18} {'Overload Dur':<16} {'Cost':<12} {'Unmet Energy'}")
    print("-" * 80)
    for cond, m in metrics.items():
        print(f"{cond:<30} {m['peak_true_load_kw']:>12.2f} kW {m['overload_minutes']:>12.1f} min {m['total_cost_usd']:>10.2f} $ {m['unmet_energy_kwh']:>12.2f} kWh")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
