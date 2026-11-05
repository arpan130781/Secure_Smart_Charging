"""Run baseline uncontrolled EV charging simulation and plot feeder load.

Simulates 24 hours at 15-minute resolution with 20-50 EVs charging in an uncontrolled manner
(each EV charges at max_rate_kw immediately on arrival until its requested energy is delivered).

Complies strictly with Rule 7:
Plain matplotlib, no decorative styling, labeled axes with units.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from ev_sched.models import Feeder
from ev_sched.simulator import generate_synthetic_sessions, run_uncontrolled_charging


def generate_diurnal_base_load(
    num_steps: int,
    nominal_kw: float,
    peak_kw: float,
    valley_kw: float,
    rng: np.random.Generator,
) -> list[float]:
    """Generate realistic diurnal background non-EV grid load profile [kW]."""
    profile: list[float] = []
    midpoint_kw = (peak_kw + valley_kw) / 2.0
    amplitude_kw = (peak_kw - valley_kw) / 2.0

    for step in range(num_steps):
        hour = (step * 15.0 / 60.0) % 24.0
        # Peak around 18:00 (6 PM), valley at 04:00 (4 AM)
        phase_rad = 2.0 * np.pi * (hour - 16.0) / 24.0
        diurnal_kw = midpoint_kw + amplitude_kw * np.sin(phase_rad)
        variation_kw = 2.0 * np.sin(2.0 * np.pi * hour / 8.0)
        noise_kw = rng.normal(0.0, 0.4)
        profile.append(float(max(10.0, diurnal_kw + variation_kw + noise_kw)))

    return profile


def plot_uncontrolled_baseline(
    config_path: Path = Path("experiments/baseline_config.yaml"),
    output_png: Path = Path("results/baseline_load.png"),
) -> None:
    """Run uncontrolled charging from YAML config and save plot."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)

    # 1. Generate base load profile
    num_steps = int(config["feeder"]["num_steps"])
    base_load = generate_diurnal_base_load(
        num_steps=num_steps,
        nominal_kw=config["feeder"]["nominal_base_kw"],
        peak_kw=config["feeder"]["peak_base_kw"],
        valley_kw=config["feeder"]["valley_base_kw"],
        rng=rng,
    )

    feeder = Feeder(
        transformer_limit_kw=float(config["feeder"]["transformer_limit_kw"]),
        base_load_profile_kw=base_load,
    )

    # 2. Generate synthetic sessions (20-50 EVs)
    sessions = generate_synthetic_sessions(config, rng)

    # 3. Simulate uncontrolled baseline charging
    result = run_uncontrolled_charging(
        feeder=feeder,
        sessions=sessions,
        time_step_min=int(config["feeder"]["time_step_minutes"]),
        num_steps=num_steps,
    )

    # 4. Generate plain matplotlib plot
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Base load
    ax.plot(
        result.step_times_hours,
        result.base_load_kw,
        color="black",
        linestyle="--",
        label="Non-EV Base Load [kW]",
    )

    # Total feeder load (base + uncontrolled EVs)
    ax.plot(
        result.step_times_hours,
        result.total_feeder_load_kw,
        color="red",
        linestyle="-",
        label="Total Feeder Load (Uncontrolled) [kW]",
    )

    # Transformer capacity limit
    ax.axhline(
        result.transformer_limit_kw,
        color="crimson",
        linestyle=":",
        linewidth=2,
        label=f"Transformer Limit ({result.transformer_limit_kw:.1f} kW)",
    )

    # Highlight overload regions where total load exceeds transformer capacity
    overload_mask = np.array(result.total_feeder_load_kw) > result.transformer_limit_kw
    if np.any(overload_mask):
        ax.fill_between(
            result.step_times_hours,
            result.transformer_limit_kw,
            result.total_feeder_load_kw,
            where=overload_mask,
            color="red",
            alpha=0.3,
            label="Transformer Overload Violation",
        )

    ax.set_title("Uncontrolled EV Charging Feeder Load vs Transformer Limit")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Active Power [kW]")
    ax.set_xlim(0.0, 24.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close(fig)

    # Print summary statistics
    peak_load = max(result.total_feeder_load_kw)
    peak_overload = max(result.overload_kw)
    total_requested = sum(s.energy_req_kwh for s in result.sessions)
    total_delivered = sum(s.delivered_energy_kwh for s in result.sessions)
    print("Baseline Simulation Run Completed:")
    print(f"  Fleet size: {len(result.sessions)} EVs")
    print(f"  Transformer limit: {result.transformer_limit_kw:.1f} kW")
    print(f"  Peak feeder load: {peak_load:.2f} kW")
    print(f"  Peak overload: {peak_overload:.2f} kW")
    print(f"  Total energy requested: {total_requested:.2f} kWh")
    print(f"  Total energy delivered: {total_delivered:.2f} kWh")
    print(f"  Figure saved to: {output_png}")


if __name__ == "__main__":
    plot_uncontrolled_baseline()
