"""Compare uncontrolled vs MPC scheduled EV charging across 20 random seeds.

Outputs:
- CSV results table: results/baseline_vs_mpc.csv
- Plain matplotlib figure: results/baseline_vs_mpc.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from ev_sched.hal import SimEVSE, SimMeter
from ev_sched.metrics import compute_run_metrics
from ev_sched.models import Feeder, Tariff
from ev_sched.scheduler import run_mpc_charging_loop
from ev_sched.simulator import (
    ChargingStationSimulator,
    generate_synthetic_sessions,
    run_uncontrolled_charging,
)
from experiments.run_baseline import generate_diurnal_base_load


def create_tou_tariff(num_steps: int = 96) -> Tariff:
    """Create realistic 24-hour Time-of-Use (TOU) electricity pricing schedule.

    - Off-peak (00:00 - 07:00, 22:00 - 24:00): $0.12 / kWh
    - Mid-peak (07:00 - 16:00, 20:00 - 22:00): $0.20 / kWh
    - On-peak  (16:00 - 20:00):                $0.38 / kWh
    """
    prices: list[float] = []
    for step in range(num_steps):
        hour = (step * 15.0 / 60.0) % 24.0
        if 16.0 <= hour < 20.0:
            prices.append(0.38)
        elif (7.0 <= hour < 16.0) or (20.0 <= hour < 22.0):
            prices.append(0.20)
        else:
            prices.append(0.12)
    return Tariff(price_per_kwh_per_15min=prices)


def run_comparison_experiment(
    config_path: Path = Path("experiments/baseline_config.yaml"),
    num_seeds: int = 20,
    results_dir: Path = Path("results"),
) -> None:
    """Execute comparative benchmark across 20 random seeds."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "baseline_vs_mpc.csv"
    png_path = results_dir / "baseline_vs_mpc.png"

    tariff = create_tou_tariff(num_steps=96)
    limit_kw = float(config["feeder"]["transformer_limit_kw"])

    seeds = [100 + i for i in range(num_seeds)]
    records: list[dict] = []

    print(f"Running comparison over {num_seeds} seeds...")

    for idx, seed in enumerate(seeds, 1):
        rng = np.random.default_rng(seed)

        # 1. Generate base load and sessions
        base_load = generate_diurnal_base_load(
            num_steps=96,
            nominal_kw=config["feeder"]["nominal_base_kw"],
            peak_kw=config["feeder"]["peak_base_kw"],
            valley_kw=config["feeder"]["valley_base_kw"],
            rng=rng,
        )
        feeder = Feeder(transformer_limit_kw=limit_kw, base_load_profile_kw=base_load)

        # Generate fresh sessions for uncontrolled run
        rng_sessions = np.random.default_rng(seed)
        sessions_uncontrolled = generate_synthetic_sessions(config, rng_sessions)

        # Duplicate sessions for MPC run
        rng_sessions_mpc = np.random.default_rng(seed)
        sessions_mpc = generate_synthetic_sessions(config, rng_sessions_mpc)

        # 2. Run Uncontrolled Baseline
        uncontrolled_res = run_uncontrolled_charging(
            feeder=feeder,
            sessions=sessions_uncontrolled,
            time_step_min=15,
            num_steps=96,
        )
        m_uncontrolled = compute_run_metrics(
            feeder_load_kw=uncontrolled_res.total_feeder_load_kw,
            ev_power_kw=uncontrolled_res.ev_load_kw,
            sessions=uncontrolled_res.sessions,
            tariff=tariff,
            transformer_limit_kw=limit_kw,
            time_step_min=15,
        )

        # 3. Run MPC Scheduled Charging Loop
        sim = ChargingStationSimulator(
            feeder=feeder,
            sessions=sessions_mpc,
            time_step_min=15,
            rng=np.random.default_rng(seed),
        )
        meter = SimMeter(sim)
        evse = SimEVSE(sim)

        mpc_res = run_mpc_charging_loop(
            feeder=feeder,
            sessions=sessions_mpc,
            tariff=tariff,
            meter=meter,
            evse=evse,
            simulator=sim,
            time_step_min=15,
            num_steps=96,
            horizon_steps=16,
        )
        m_mpc = compute_run_metrics(
            feeder_load_kw=[b + e for b, e in zip(mpc_res.base_load_kw, mpc_res.total_ev_power_kw)],
            ev_power_kw=mpc_res.total_ev_power_kw,
            sessions=mpc_res.sessions,
            tariff=tariff,
            transformer_limit_kw=limit_kw,
            time_step_min=15,
        )

        row = {
            "seed": seed,
            "uncontrolled_peak_kw": m_uncontrolled.peak_load_kw,
            "mpc_peak_kw": m_mpc.peak_load_kw,
            "uncontrolled_overload_min": m_uncontrolled.overload_minutes,
            "mpc_overload_min": m_mpc.overload_minutes,
            "uncontrolled_cost_usd": m_uncontrolled.total_cost_usd,
            "mpc_cost_usd": m_mpc.total_cost_usd,
            "uncontrolled_unmet_kwh": m_uncontrolled.unmet_energy_kwh,
            "mpc_unmet_kwh": m_mpc.unmet_energy_kwh,
        }
        records.append(row)
        print(
            f"  Seed {seed} ({idx}/{num_seeds}): "
            f"Peak load: {m_uncontrolled.peak_load_kw:.1f} -> {m_mpc.peak_load_kw:.1f} kW | "
            f"Overload: {m_uncontrolled.overload_minutes:.0f} -> {m_mpc.overload_minutes:.0f} min | "
            f"Cost: ${m_uncontrolled.total_cost_usd:.2f} -> ${m_mpc.total_cost_usd:.2f}"
        )

    # 4. Save results to CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"Saved comparative metrics across 20 seeds to: {csv_path}")

    # 5. Generate plain matplotlib comparison plots
    plot_comparison(records, limit_kw, png_path)
    print(f"Saved comparison plot to: {png_path}")


def plot_comparison(records: list[dict], limit_kw: float, output_png: Path) -> None:
    """Generate plain matplotlib plot comparing uncontrolled vs MPC across 20 seeds."""
    seed_indices = np.arange(len(records)) + 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Subplot 1: Peak Feeder Load [kW]
    ax1 = axes[0, 0]
    ax1.plot(
        seed_indices,
        [r["uncontrolled_peak_kw"] for r in records],
        marker="o",
        color="red",
        linestyle="-",
        label="Uncontrolled",
    )
    ax1.plot(
        seed_indices,
        [r["mpc_peak_kw"] for r in records],
        marker="s",
        color="blue",
        linestyle="--",
        label="MPC Scheduled",
    )
    ax1.axhline(
        limit_kw,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"Transformer Limit ({limit_kw:.0f} kW)",
    )
    ax1.set_title("Peak Feeder Load [kW]")
    ax1.set_xlabel("Seed Index")
    ax1.set_ylabel("Active Power [kW]")
    ax1.set_xticks(seed_indices)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    # Subplot 2: Overload Duration [minutes]
    ax2 = axes[0, 1]
    ax2.bar(
        seed_indices - 0.2,
        [r["uncontrolled_overload_min"] for r in records],
        width=0.4,
        color="red",
        label="Uncontrolled",
    )
    ax2.bar(
        seed_indices + 0.2,
        [r["mpc_overload_min"] for r in records],
        width=0.4,
        color="blue",
        label="MPC Scheduled",
    )
    ax2.set_title("Overload Duration [minutes]")
    ax2.set_xlabel("Seed Index")
    ax2.set_ylabel("Duration [minutes]")
    ax2.set_xticks(seed_indices)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    # Subplot 3: Electricity Cost [$]
    ax3 = axes[1, 0]
    ax3.plot(
        seed_indices,
        [r["uncontrolled_cost_usd"] for r in records],
        marker="o",
        color="red",
        linestyle="-",
        label="Uncontrolled",
    )
    ax3.plot(
        seed_indices,
        [r["mpc_cost_usd"] for r in records],
        marker="s",
        color="blue",
        linestyle="--",
        label="MPC Scheduled",
    )
    ax3.set_title("Total Electricity Cost [$]")
    ax3.set_xlabel("Seed Index")
    ax3.set_ylabel("Cost [$]")
    ax3.set_xticks(seed_indices)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper left")

    # Subplot 4: Unmet Energy [kWh]
    ax4 = axes[1, 1]
    ax4.bar(
        seed_indices - 0.2,
        [r["uncontrolled_unmet_kwh"] for r in records],
        width=0.4,
        color="red",
        label="Uncontrolled",
    )
    ax4.bar(
        seed_indices + 0.2,
        [r["mpc_unmet_kwh"] for r in records],
        width=0.4,
        color="blue",
        label="MPC Scheduled",
    )
    ax4.set_title("Unmet Energy [kWh]")
    ax4.set_xlabel("Seed Index")
    ax4.set_ylabel("Energy [kWh]")
    ax4.set_xticks(seed_indices)
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend(loc="upper left")

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run_comparison_experiment()
