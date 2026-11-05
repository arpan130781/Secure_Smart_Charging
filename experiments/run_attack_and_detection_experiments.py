"""Execute Tasks 11 & 14: Attack Impact Analysis and Cyber Detector Evaluation.

Simulates 20 seeds across 4 conditions:
1. No Attack (Baseline MPC)
2. Constant Bias FDI Attack (-35 kW)
3. Ramp FDI Attack (0 -> -35 kW)
4. Replay FDI Attack (replaying night valley)

Evaluates:
- Attack Impact on Grid Safety & QoS -> results/attack_impact.csv & results/attack_impact.png
- Detector 1 (CUSUM) & Detector 2 (Isolation Forest) -> results/detection_metrics.csv
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import numpy as np
import matplotlib.pyplot as plt

from ev_sched.models import EVSession, Feeder, Tariff
from ev_sched.simulator import (
    ChargingStationSimulator,
    generate_synthetic_sessions,
)
from ev_sched.hal import SimEVSE, SimMeter
from ev_sched.scheduler import solve_charging_schedule
from ev_sched.attacks import create_attacked_meter
from ev_sched.detector import CUSUMDetector, IsolationForestDetector
from ev_sched.metrics import compute_run_metrics
from experiments.run_baseline import generate_diurnal_base_load
from experiments.compare_baseline_vs_mpc import create_tou_tariff


def train_isolation_forest_on_clean_runs(
    config: dict,
    feeder_limit_kw: float,
    tariff: Tariff,
    num_train_seeds: int = 5,
) -> IsolationForestDetector:
    """Train Isolation Forest on attack-free operational telemetry."""
    print("Training Isolation Forest Detector on attack-free baseline runs...")
    detector = IsolationForestDetector(contamination=0.01, random_state=42)
    feature_rows: List[np.ndarray] = []

    for seed in range(500, 500 + num_train_seeds):
        rng = np.random.default_rng(seed)
        base_load = generate_diurnal_base_load(
            num_steps=96,
            nominal_kw=config["feeder"]["nominal_base_kw"],
            peak_kw=config["feeder"]["peak_base_kw"],
            valley_kw=config["feeder"]["valley_base_kw"],
            rng=rng,
        )
        feeder = Feeder(transformer_limit_kw=feeder_limit_kw, base_load_profile_kw=base_load)
        sessions = generate_synthetic_sessions(config, rng)
        sim = ChargingStationSimulator(feeder, sessions, time_step_min=15, rng=rng)
        meter = SimMeter(sim, noise_std_kw=0.3)
        evse = SimEVSE(sim)

        prev_r = 0.0
        for step in range(96):
            t_min = step * 15
            meas_kw = meter.read_feeder_kw(t_min)
            cmd_kw_total = sim.get_total_ev_power_kw()
            expected_kw = base_load[step] + cmd_kw_total
            feat = detector.extract_features(meas_kw, expected_kw, prev_residual_kw=prev_r)
            prev_r = float(feat[0])
            feature_rows.append(feat)
            sim.step()

    X_train = np.vstack(feature_rows)
    detector.fit_from_attack_free_runs(X_train)
    print(f"Isolation Forest trained successfully on {len(X_train)} clean operational steps.")
    return detector


def run_mpc_with_monitoring(
    feeder: Feeder,
    sessions: List[EVSession],
    tariff: Tariff,
    attack_config: dict | None,
    iso_detector: IsolationForestDetector,
    seed: int,
    time_step_min: int = 15,
    num_steps: int = 96,
    horizon_steps: int = 16,
) -> Tuple[List[float], List[float], List[bool], List[bool], List[bool]]:
    """Run MPC loop under optional attack and monitor with both detectors.

    Returns
    -------
    Tuple containing:
    - true_feeder_load_kw : List[float]
    - reported_feeder_kw  : List[float]
    - is_attack_mask      : List[bool]
    - cusum_alarms        : List[bool]
    - iso_alarms          : List[bool]
    """
    dt_hours = time_step_min / 60.0
    rng = np.random.default_rng(seed)

    sim = ChargingStationSimulator(feeder=feeder, sessions=sessions, time_step_min=time_step_min, rng=rng)
    base_meter = SimMeter(sim, noise_std_kw=0.3)
    evse = SimEVSE(sim)

    # Wrap meter if attack configured
    if attack_config is not None:
        meter = create_attacked_meter(base_meter, attack_config)
    else:
        meter = base_meter

    cusum = CUSUMDetector(threshold=18.0, slack_kw=1.5, instant_threshold_kw=20.0)
    iso_detector.reset()

    true_feeder_kw: List[float] = []
    reported_kw: List[float] = []
    attack_mask: List[bool] = []
    cusum_alarms: List[bool] = []
    iso_alarms: List[bool] = []

    base_profile = list(feeder.base_load_profile_kw)
    price_profile = list(tariff.price_per_kwh_per_15min)

    for step in range(num_steps):
        t_min = step * time_step_min

        # Check ground truth attack state
        is_attack = False
        if attack_config is not None and hasattr(meter, "is_attack_active"):
            is_attack = meter.is_attack_active(t_min)
        attack_mask.append(is_attack)

        # 1. Read meter via HAL (under attack if active)
        meas_kw = meter.read_feeder_kw(t_min)
        reported_kw.append(meas_kw)

        curr_base_kw = base_profile[step]

        # In SCADA/EMS systems, background base load is inferred from feeder telemetry:
        # P_base_inferred = max(0, meas_feeder_kw - ev_submetered_kw).
        # Under False Data Injection, this offset directly corrupts the scheduler's lookahead headroom!
        current_ev_measured_kw = sim.get_total_ev_power_kw()
        inferred_current_base = max(0.0, meas_kw - current_ev_measured_kw)
        telemetry_offset_kw = inferred_current_base - curr_base_kw

        # 2. Re-solve MPC over remaining horizon using poisoned telemetry offset
        rem_steps = num_steps - step
        plan_steps = min(horizon_steps, rem_steps)
        rem_base = [
            max(0.0, b + telemetry_offset_kw)
            for b in base_profile[step : step + plan_steps]
        ]
        rem_feeder = Feeder(transformer_limit_kw=feeder.transformer_limit_kw, base_load_profile_kw=rem_base)
        rem_prices = price_profile[step : step + plan_steps]
        rem_tariff = Tariff(price_per_kwh_per_15min=rem_prices)

        # Construct shifted sessions
        active_rem_sessions: List[EVSession] = []
        for s in sessions:
            arr_rel = max(0, s.arrival_min - t_min)
            dep_rel = max(1, s.departure_min - t_min)
            rem_energy = (
                s.remaining_energy_req_kwh
                if (s.is_connected(t_min) or (s.arrival_min >= t_min and s.arrival_min < t_min + plan_steps * time_step_min))
                else 0.0
            )
            shifted_s = EVSession(
                arrival_min=arr_rel,
                departure_min=dep_rel,
                energy_req_kwh=rem_energy,
                max_rate_kw=s.max_rate_kw,
                session_id=s.session_id,
            )
            active_rem_sessions.append(shifted_s)

        opt_res = solve_charging_schedule(
            sessions=active_rem_sessions,
            feeder=rem_feeder,
            tariff=rem_tariff,
            time_step_min=time_step_min,
            num_steps=plan_steps,
        )

        # 3. Apply first step setpoints
        step_ev_cmd = 0.0
        for i, s in enumerate(sessions):
            cmd_kw = float(opt_res.power_schedule_kw[i, 0]) if (opt_res.is_success and i < opt_res.power_schedule_kw.shape[0]) else 0.0
            evse.set_setpoint_kw(s.session_id, cmd_kw)
            actual_kw = evse.read_session_kw(s.session_id)
            step_ev_cmd += actual_kw

        # Expected feeder load = base load forecast + aggregate EV power currently commanded/active
        expected_feeder_kw = curr_base_kw + current_ev_measured_kw

        # 4. Update detectors
        c_alert = cusum.update(reported_feeder_kw=meas_kw, expected_feeder_kw=expected_feeder_kw)
        cusum_alarms.append(c_alert.is_alarm)

        i_alert = iso_detector.update(reported_feeder_kw=meas_kw, expected_feeder_kw=expected_feeder_kw)
        iso_alarms.append(i_alert.is_alarm)

        # Ground truth true feeder power
        true_feeder_kw.append(sim.get_feeder_power_kw(t_min))

        # Advance simulator
        sim.step()

    return true_feeder_kw, reported_kw, attack_mask, cusum_alarms, iso_alarms


def run_all_experiments(
    config_path: Path = Path("experiments/baseline_config.yaml"),
    results_dir: Path = Path("results"),
    num_seeds: int = 20,
) -> None:
    """Run Tasks 11 & 14 end-to-end and export results."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results_dir.mkdir(parents=True, exist_ok=True)
    tariff = create_tou_tariff(96)
    limit_kw = float(config["feeder"]["transformer_limit_kw"])

    # 1. Train Isolation Forest
    iso_detector = train_isolation_forest_on_clean_runs(config, limit_kw, tariff, num_train_seeds=5)

    # 2. Define Attack Configurations
    attacks = {
        "no_attack": None,
        "constant_bias": {
            "attack_type": "constant_bias",
            "start_min": 480,       # 08:00 AM
            "duration_min": 360,     # 6 hours
            "magnitude_kw": 35.0,
        },
        "ramp": {
            "attack_type": "ramp",
            "start_min": 480,       # 08:00 AM
            "duration_min": 360,
            "magnitude_kw": 35.0,
        },
        "replay": {
            "attack_type": "replay",
            "start_min": 480,       # 08:00 AM
            "duration_min": 360,
            "replay_source_start_min": 60,  # Replay early morning 01:00 AM valley
        },
    }

    seeds = [200 + i for i in range(num_seeds)]
    impact_records: List[dict] = []

    # Detection metrics tracking: { (detector, attack): [ (y_true, y_pred, latency) ] }
    eval_data: Dict[Tuple[str, str], Dict[str, list]] = {
        (det, atk): {"y_true": [], "y_pred": [], "latencies": []}
        for det in ["CUSUM", "IsolationForest"]
        for atk in ["constant_bias", "ramp", "replay"]
    }

    # Store sample trajectory for plotting (from seed 200)
    plot_trajectories: Dict[str, List[float]] = {}

    print(f"\nSimulating MPC under 4 conditions across {num_seeds} seeds...")

    for atk_name, atk_cfg in attacks.items():
        print(f"--> Running condition: {atk_name}...")
        for seed in seeds:
            rng = np.random.default_rng(seed)
            base_load = generate_diurnal_base_load(
                num_steps=96,
                nominal_kw=config["feeder"]["nominal_base_kw"],
                peak_kw=config["feeder"]["peak_base_kw"],
                valley_kw=config["feeder"]["valley_base_kw"],
                rng=rng,
            )
            feeder = Feeder(transformer_limit_kw=limit_kw, base_load_profile_kw=base_load)

            rng_ev = np.random.default_rng(seed)
            sessions = generate_synthetic_sessions(config, rng_ev)

            true_load, reported_load, attack_mask, cusum_alerts, iso_alerts = run_mpc_with_monitoring(
                feeder=feeder,
                sessions=sessions,
                tariff=tariff,
                attack_config=atk_cfg,
                iso_detector=iso_detector,
                seed=seed,
            )

            if seed == seeds[0]:
                plot_trajectories[atk_name] = true_load

            # Calculate Task 11 impact metrics on TRUE feeder load
            m = compute_run_metrics(
                feeder_load_kw=true_load,
                ev_power_kw=[max(0.0, true - base) for true, base in zip(true_load, base_load)],
                sessions=sessions,
                tariff=tariff,
                transformer_limit_kw=limit_kw,
                time_step_min=15,
            )

            impact_records.append({
                "seed": seed,
                "attack_type": atk_name,
                "peak_true_load_kw": round(m.peak_load_kw, 2),
                "overload_minutes": round(m.overload_minutes, 1),
                "total_cost_usd": round(m.total_cost_usd, 2),
                "unmet_energy_kwh": round(m.unmet_energy_kwh, 2),
            })

            # Record detection metrics if under attack
            if atk_name in ["constant_bias", "ramp", "replay"]:
                # Detection delay calculation
                first_atk_step = 480 // 15  # step 32
                c_delay = None
                i_delay = None

                for step_idx in range(first_atk_step, first_atk_step + (360 // 15)):
                    if cusum_alerts[step_idx] and c_delay is None:
                        c_delay = (step_idx - first_atk_step) * 15
                    if iso_alerts[step_idx] and i_delay is None:
                        i_delay = (step_idx - first_atk_step) * 15

                eval_data[("CUSUM", atk_name)]["y_true"].extend(attack_mask)
                eval_data[("CUSUM", atk_name)]["y_pred"].extend(cusum_alerts)
                if c_delay is not None:
                    eval_data[("CUSUM", atk_name)]["latencies"].append(c_delay)

                eval_data[("IsolationForest", atk_name)]["y_true"].extend(attack_mask)
                eval_data[("IsolationForest", atk_name)]["y_pred"].extend(iso_alerts)
                if i_delay is not None:
                    eval_data[("IsolationForest", atk_name)]["latencies"].append(i_delay)

    # 3. Export Task 11: results/attack_impact.csv
    impact_csv_path = results_dir / "attack_impact.csv"
    with open(impact_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(impact_records[0].keys()))
        writer.writeheader()
        for r in impact_records:
            writer.writerow(r)
    print(f"Saved Task 11 impact results to: {impact_csv_path}")

    # 4. Generate Task 11 Plot: results/attack_impact.png
    plot_attack_impact(plot_trajectories, limit_kw, results_dir / "attack_impact.png")
    print(f"Saved Task 11 plot to: {results_dir / 'attack_impact.png'}")

    # 5. Compute & Export Task 14: results/detection_metrics.csv
    detection_rows: List[dict] = []
    for (det_name, atk_name), data in eval_data.items():
        y_true = np.array(data["y_true"], dtype=int)
        y_pred = np.array(data["y_pred"], dtype=int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # False alarm rate per day: FP divided by number of normal days evaluated
        # (Total unattacked steps across 20 runs = 20 * (96 - 24) = 1440 steps = 15 days of clean data)
        clean_days = float((len(y_true) - np.sum(y_true)) / 96.0)
        far_per_day = float(fp / clean_days) if clean_days > 0 else 0.0

        latencies = data["latencies"]
        mean_delay_min = float(np.mean(latencies)) if latencies else float("nan")

        detection_rows.append({
            "detector": det_name,
            "attack_type": atk_name,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "false_alarms_per_day": round(far_per_day, 2),
            "detection_delay_minutes": round(mean_delay_min, 1) if not np.isnan(mean_delay_min) else "N/A",
        })

    detection_csv_path = results_dir / "detection_metrics.csv"
    with open(detection_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detection_rows[0].keys()))
        writer.writeheader()
        for r in detection_rows:
            writer.writerow(r)
    print(f"Saved Task 14 detection metrics to: {detection_csv_path}")

    # Print Executive Summary Tables
    print("\n" + "=" * 80)
    print("TASK 11: ATTACK IMPACT EXECUTIVE SUMMARY (AVERAGES OVER 20 SEEDS)")
    print("=" * 80)
    print(f"{'Condition':<18} {'Peak True Load [kW]':>22} {'Overload Dur [min]':>22} {'Cost [$]':>14} {'Unmet [kWh]':>14}")
    print("-" * 90)
    for atk_name in ["no_attack", "constant_bias", "ramp", "replay"]:
        recs = [r for r in impact_records if r["attack_type"] == atk_name]
        avg_peak = np.mean([r["peak_true_load_kw"] for r in recs])
        avg_ovld = np.mean([r["overload_minutes"] for r in recs])
        avg_cost = np.mean([r["total_cost_usd"] for r in recs])
        avg_unmet = np.mean([r["unmet_energy_kwh"] for r in recs])
        print(f"{atk_name:<18} {avg_peak:>20.2f} kW {avg_ovld:>20.1f} min ${avg_cost:>12.2f} {avg_unmet:>12.2f} kWh")

    print("\n" + "=" * 80)
    print("TASK 14: DETECTOR EVALUATION EXECUTIVE SUMMARY")
    print("=" * 80)
    print(f"{'Detector':<18} {'Attack Type':<16} {'Precision':>12} {'Recall':>10} {'FAR [/day]':>14} {'Delay [min]':>14}")
    print("-" * 84)
    for r in detection_rows:
        print(f"{r['detector']:<18} {r['attack_type']:<16} {r['precision']:>12} {r['recall']:>10} {r['false_alarms_per_day']:>14} {r['detection_delay_minutes']:>14}")
    print("=" * 80)


def plot_attack_impact(trajectories: Dict[str, List[float]], limit_kw: float, output_png: Path) -> None:
    """Generate plain matplotlib plot comparing true feeder load across attack conditions."""
    hours = np.arange(96) * 15.0 / 60.0

    fig, ax = plt.subplots(figsize=(11, 6))

    if "no_attack" in trajectories:
        ax.plot(hours, trajectories["no_attack"], color="black", linestyle="-", linewidth=2.0, label="No Attack (MPC Baseline)")
    if "constant_bias" in trajectories:
        ax.plot(hours, trajectories["constant_bias"], color="red", linestyle="--", linewidth=1.8, label="Constant Bias Attack (-35 kW)")
    if "ramp" in trajectories:
        ax.plot(hours, trajectories["ramp"], color="orange", linestyle="-.", linewidth=1.8, label="Ramp Attack (0 to -35 kW)")
    if "replay" in trajectories:
        ax.plot(hours, trajectories["replay"], color="purple", linestyle=":", linewidth=2.0, label="Replay Attack (Early Morning Valley)")

    ax.axhline(limit_kw, color="crimson", linestyle=":", linewidth=2.2, label=f"Transformer Limit ({limit_kw:.0f} kW)")

    # Highlight attack window (08:00 to 14:00, hours 8 to 14)
    ax.axvspan(8.0, 14.0, color="gray", alpha=0.15, label="Attack Window (08:00 - 14:00)")

    ax.set_title("True Feeder Active Power Load Under False Data Injection Attacks")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("True Feeder Load [kW]")
    ax.set_xlim(0.0, 24.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run_all_experiments()
