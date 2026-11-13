"""Tests for user tasks 1-8.

Each test in this file is written to fail BEFORE the corresponding task fix is applied.
"""

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from ev_sched.hal import SimMeter
from ev_sched.models import EVSession, Feeder


# ==============================================================================
# Task 1: SimMeter independent RNG derived from run seed
# ==============================================================================
def test_task1_sim_meter_rng_isolation_in_experiments():
    """SimMeter must be instantiated with seed/rng derived from run seed, not default 42."""
    from experiments.compare_baseline_vs_mpc import create_tou_tariff
    from experiments.run_attack_and_detection_experiments import run_mpc_with_monitoring

    feeder = Feeder(transformer_limit_kw=120.0, base_load_profile_kw=[30.0] * 96)
    tariff = create_tou_tariff(96)
    sessions = [
        EVSession(arrival_min=0, departure_min=120, energy_req_kwh=10.0, max_rate_kw=7.0, session_id="EV1")
    ]

    from ev_sched.detector import IsolationForestDetector
    iso = IsolationForestDetector()

    # Intercept SimMeter creation in run_mpc_with_monitoring to inspect its seed/RNG
    passed_args: list[dict] = []
    orig_init = SimMeter.__init__

    def tracked_init(self, simulator, noise_std_kw=0.0, rng=None, seed=42):
        passed_args.append({"rng": rng, "seed": seed})
        orig_init(self, simulator, noise_std_kw=noise_std_kw, rng=rng, seed=seed)

    with patch.object(SimMeter, "__init__", tracked_init):
        run_mpc_with_monitoring(
            feeder=feeder,
            sessions=sessions,
            tariff=tariff,
            attack_config=None,
            iso_detector=iso,
            seed=200,
            num_steps=4,
        )
        run_mpc_with_monitoring(
            feeder=feeder,
            sessions=sessions,
            tariff=tariff,
            attack_config=None,
            iso_detector=iso,
            seed=201,
            num_steps=4,
        )

    assert len(passed_args) >= 2
    # The caller must pass an independent rng or seed derived from the run seed, NOT default rng=None & seed=42
    assert not (passed_args[0]["rng"] is None and passed_args[0]["seed"] == 42), (
        "SimMeter in seed=200 was called with default rng=None and seed=42!"
    )
    assert not (passed_args[1]["rng"] is None and passed_args[1]["seed"] == 42), (
        "SimMeter in seed=201 was called with default rng=None and seed=42!"
    )


# ==============================================================================
# Task 2: Training Isolation Forest on clean runs must see EV load
# ==============================================================================
def test_task2_train_isolation_forest_sees_ev_load():
    """train_isolation_forest_on_clean_runs must dispatch EV setpoints so training features see EV load."""
    import ev_sched.scheduler as sched
    from ev_sched.detector import IsolationForestDetector
    from experiments.compare_baseline_vs_mpc import create_tou_tariff
    from experiments.run_attack_and_detection_experiments import (
        train_isolation_forest_on_clean_runs,
    )

    config_path = Path("experiments/baseline_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tariff = create_tou_tariff(96)
    
    # We patch IsolationForestDetector.fit_from_attack_free_runs to inspect X_train
    fitted_X: list[np.ndarray] = []
    orig_fit = IsolationForestDetector.fit_from_attack_free_runs

    def intercept_fit(self, X):
        fitted_X.append(X)
        return orig_fit(self, X)

    recorded_mpc_results: list[sched.MPCLoopResult] = []
    orig_run_mpc = sched.run_mpc_charging_loop

    def intercept_run_mpc(*args, **kwargs):
        res = orig_run_mpc(*args, **kwargs)
        recorded_mpc_results.append(res)
        return res

    with patch.object(sched, "run_mpc_charging_loop", intercept_run_mpc), \
         patch.object(IsolationForestDetector, "fit_from_attack_free_runs", intercept_fit):
        train_isolation_forest_on_clean_runs(
            config=config,
            feeder_limit_kw=120.0,
            tariff=tariff,
            num_train_seeds=1,
        )

    # 1. Verify that real MPC loop was executed and recorded
    assert len(recorded_mpc_results) == 1, "run_mpc_charging_loop was not called during training!"
    mpc_res = recorded_mpc_results[0]

    # 2. Assert from recorded MPC results that EV charging power was actually positive during training
    positive_ev_power = [p for p in mpc_res.total_ev_power_kw if p > 0.0]
    assert len(positive_ev_power) > 0, (
        f"Recorded MPC results contained no positive EV charging power during training! "
        f"total_ev_power_kw={mpc_res.total_ev_power_kw}"
    )
    assert max(mpc_res.total_ev_power_kw) > 0.0, "Peak EV charging power must be positive during training"
    assert sum(mpc_res.total_ev_power_kw) > 0.0, "Total delivered EV energy must be positive during training"

    # 3. Assert from training features that genuine feature vectors were extracted and passed to fit
    assert len(fitted_X) == 1
    X_train = fitted_X[0]
    assert X_train.shape[0] == 96
    assert X_train.shape[1] == 3


# ==============================================================================
# Task 3: Forecast error option in simulator with AR(1) noise
# ==============================================================================
def test_task3_forecast_error_option_ar1():
    """Simulator must provide generate_base_forecast_with_ar1_error with rho=0.9 and configurable std."""
    from ev_sched.simulator import generate_base_forecast_with_ar1_error

    true_base = [30.0] * 96
    # Test zero std produces exact truth
    exact = generate_base_forecast_with_ar1_error(true_base, rho=0.9, std_kw=0.0)
    assert exact == true_base

    # Test default std=3.0 produces stationary AR(1) variations
    rng = np.random.default_rng(42)
    noisy = generate_base_forecast_with_ar1_error(true_base, rho=0.9, std_kw=3.0, rng=rng)
    assert len(noisy) == 96
    assert noisy != true_base
    errors = np.array(noisy) - np.array(true_base)
    # Check empirical standard deviation is close to 3.0 kW
    assert 1.5 < np.std(errors) < 4.5

    # Input validation checks:
    # 1. rho < 0 must raise ValueError
    with pytest.raises(ValueError, match="rho must satisfy 0 <= rho < 1"):
        generate_base_forecast_with_ar1_error(true_base, rho=-0.5)

    with pytest.raises(ValueError, match="rho must satisfy 0 <= rho < 1"):
        generate_base_forecast_with_ar1_error(true_base, rho=-0.01)

    # 2. rho >= 1 must raise ValueError
    with pytest.raises(ValueError, match="rho must satisfy 0 <= rho < 1"):
        generate_base_forecast_with_ar1_error(true_base, rho=1.0)

    with pytest.raises(ValueError, match="rho must satisfy 0 <= rho < 1"):
        generate_base_forecast_with_ar1_error(true_base, rho=1.5)

    # 3. std_kw < 0 must raise ValueError
    with pytest.raises(ValueError, match="std_kw must be >= 0"):
        generate_base_forecast_with_ar1_error(true_base, rho=0.9, std_kw=-1.0)

    with pytest.raises(ValueError, match="std_kw must be >= 0"):
        generate_base_forecast_with_ar1_error(true_base, rho=0.9, std_kw=-0.001)

    # 4. Non-finite values (NaN, infinity) must raise ValueError
    with pytest.raises(ValueError, match="rho must be a finite number"):
        generate_base_forecast_with_ar1_error(true_base, rho=float("nan"))

    with pytest.raises(ValueError, match="rho must be a finite number"):
        generate_base_forecast_with_ar1_error(true_base, rho=float("inf"))

    with pytest.raises(ValueError, match="std_kw must be a finite number"):
        generate_base_forecast_with_ar1_error(true_base, std_kw=float("nan"))

    with pytest.raises(ValueError, match="std_kw must be a finite number"):
        generate_base_forecast_with_ar1_error(true_base, std_kw=float("inf"))

    # 5. Valid edge cases: rho=0.0 and std_kw=0.0 are permitted
    zero_rho = generate_base_forecast_with_ar1_error(true_base, rho=0.0, std_kw=1.0, rng=rng)
    assert len(zero_rho) == 96
    zero_std = generate_base_forecast_with_ar1_error(true_base, rho=0.0, std_kw=0.0)
    assert zero_std == true_base


# ==============================================================================
# Task 4: run_mpc_with_monitoring delegates to run_mpc_charging_loop
# ==============================================================================
def test_task4_run_mpc_with_monitoring_delegates_to_run_mpc_charging_loop():
    """run_mpc_with_monitoring must not duplicate MPC logic; it must call run_mpc_charging_loop."""
    import ev_sched.scheduler as sched_mod
    from experiments.compare_baseline_vs_mpc import create_tou_tariff
    from experiments.run_attack_and_detection_experiments import run_mpc_with_monitoring

    feeder = Feeder(transformer_limit_kw=120.0, base_load_profile_kw=[30.0] * 96)
    tariff = create_tou_tariff(96)
    sessions = [
        EVSession(arrival_min=0, departure_min=120, energy_req_kwh=10.0, max_rate_kw=7.0, session_id="EV1")
    ]
    from ev_sched.detector import IsolationForestDetector
    iso = IsolationForestDetector()

    called = []
    orig_loop = sched_mod.run_mpc_charging_loop

    def mock_loop(*args, **kwargs):
        called.append(True)
        return orig_loop(*args, **kwargs)

    with patch.object(sched_mod, "run_mpc_charging_loop", side_effect=mock_loop):
        run_mpc_with_monitoring(
            feeder=feeder,
            sessions=sessions,
            tariff=tariff,
            attack_config=None,
            iso_detector=iso,
            seed=200,
            num_steps=4,
        )

    assert len(called) > 0, "run_mpc_with_monitoring did NOT call run_mpc_charging_loop!"


# ==============================================================================
# Task 5: Metrics precision=NaN when no alarms, episode-level metrics, clean-day count
# ==============================================================================
def test_task5_metrics_precision_nan_and_episode_metrics():
    """When tp + fp == 0, precision must be NaN, and episode-level metrics must be exported."""
    from experiments.run_attack_and_detection_experiments import (
        compute_detection_summary_metrics,
    )

    # Case with no alarms at all: tp=0, fp=0, fn=10
    eval_record = {
        "y_true": [1] * 10 + [0] * 86,
        "y_pred": [0] * 96,
        "latencies": [],
        "episode_outcomes": [{"detected": False}],
    }
    metrics = compute_detection_summary_metrics(eval_record, total_clean_days=50.0)
    assert math.isnan(metrics["precision"]), f"Expected NaN precision when no alarms, got {metrics['precision']}"
    assert "episode_detection_rate" in metrics
    assert "episode_misses" in metrics
    assert metrics["episode_misses"] == 1
    assert metrics["episode_detection_rate"] == 0.0


# ==============================================================================
# Task 6: Replay attack variant replaying a similar-load window
# ==============================================================================
def test_task6_similar_load_replay_attack():
    """attacks.py must provide SimilarLoadReplayAttackMeter."""
    from ev_sched.attacks import SimilarLoadReplayAttackMeter, create_attacked_meter

    class DummyMeter:
        def __init__(self):
            # profile: 0-100 min: 30kW, 100-200 min: 80kW, 200-300 min: 32kW
            pass
        def read_feeder_kw(self, t_min: int) -> float:
            if t_min < 100:
                return 30.0
            elif t_min < 200:
                return 80.0
            return 32.0

    meter = DummyMeter()
    # Populate history
    attack = SimilarLoadReplayAttackMeter(
        inner_meter=meter,
        start_min=200,
        duration_min=60,
        candidate_search_start_min=0,
        candidate_search_end_min=200,
    )
    for t in range(0, 200, 15):
        attack.read_feeder_kw(t)

    # When attack starts at t=200 (load=32kW), it should pick window around t=0..60 (load=30kW), NOT t=100..160 (load=80kW)
    replayed_val = attack.read_feeder_kw(200)
    assert abs(replayed_val - 30.0) < 5.0, f"Expected similar load ~30kW, got {replayed_val}"

    # Verify factory support
    factory_meter = create_attacked_meter(meter, {"attack_type": "similar_load_replay", "start_min": 200, "duration_min": 60})
    assert isinstance(factory_meter, SimilarLoadReplayAttackMeter)


# ==============================================================================
# Task 7: Headline cost per delivered kWh and multi-seed execution
# ==============================================================================
def test_task7_headline_cost_per_delivered_kwh_and_multiseed():
    """Headline simulation must compute cost_per_delivered_kwh_usd and support multi-seed execution."""
    from experiments.run_headline_experiment import (
        run_headline_multiseed_experiment,
        run_headline_simulation,
    )

    config_path = Path("experiments/baseline_config.yaml")
    # Single seed run must include cost_per_delivered_kwh_usd
    _, metrics = run_headline_simulation(config_path=config_path, seed=42, num_steps=16)
    assert "cost_per_delivered_kwh_usd" in metrics["normal"]

    # Multi-seed run must return mean and std across seeds
    multi_summary = run_headline_multiseed_experiment(config_path=config_path, num_seeds=2, num_steps=16)
    assert "normal" in multi_summary
    assert "cost_per_delivered_kwh_usd_mean" in multi_summary["normal"]
    assert "cost_per_delivered_kwh_usd_std" in multi_summary["normal"]


# ==============================================================================
# Task 8: README documents 0.5 kW tolerance and Isolation Forest limitation
# ==============================================================================
def test_task8_readme_documents_tolerance_and_iso_limitation():
    """README must document the 0.5 kW overload tolerance band and the Isolation Forest change-detector limitation."""
    readme_text = Path("README.md").read_text(encoding="utf-8")
    assert "0.5" in readme_text and "tolerance" in readme_text.lower(), "README does not document the 0.5 kW overload tolerance!"
    assert "change-detector" in readme_text.lower() or "transient" in readme_text.lower(), "README does not document Isolation Forest change-detector limitation!"


# ==============================================================================
# Task 10: Non-time-step-aligned session windows and partial interval modeling
# ==============================================================================
def test_task10_non_aligned_session_window_partial_intervals():
    """Verify non-time-step-aligned sessions do not charge before arrival or count energy after departure.

    Regression case: 10-minute arrival and 20-minute departure on a 15-minute step.
    - Physical dwell: 10 minutes (10 to 20 min).
    - Maximum physical energy at 7.2 kW: 7.2 * (10 / 60) = 1.2 kWh.
    - Step 0 ([0, 15) min): connected only 10..15 min (5 min, 1/3 step) -> power <= 2.4 kW, energy <= 0.6 kWh.
      Cannot charge before arrival (0..10 min).
    - Step 1 ([15, 30) min): connected only 15..20 min (5 min, 1/3 step) -> power <= 2.4 kW, energy <= 0.6 kWh.
      Cannot count energy after departure (20..30 min).
    - Step 2+ ([30, 60) min): vehicle disconnected -> power == 0.0 kW.
    """
    from ev_sched.models import Tariff
    from ev_sched.scheduler import (
        solve_charging_schedule,
        validate_schedule_constraints,
    )
    from ev_sched.simulator import (
        ChargingStationSimulator,
        run_uncontrolled_charging,
    )

    session = EVSession(
        arrival_min=10,
        departure_min=20,
        energy_req_kwh=5.0,
        max_rate_kw=7.2,
        session_id="EV_PARTIAL_TASK10",
    )
    feeder = Feeder(transformer_limit_kw=100.0, base_load_profile_kw=[20.0] * 4)
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 4)
    time_step_min = 15
    num_steps = 4
    dt_hours = 15.0 / 60.0

    res = solve_charging_schedule(
        sessions=[session],
        feeder=feeder,
        tariff=tariff,
        time_step_min=time_step_min,
        num_steps=num_steps,
    )
    assert res.is_success
    P = res.power_schedule_kw

    # Check power bounds: step 0 and 1 are bounded by 2.4 kW, steps 2 and 3 are 0.0 kW
    assert P[0, 0] <= 2.4 + 1e-4
    assert P[0, 1] <= 2.4 + 1e-4
    assert P[0, 2] == pytest.approx(0.0, abs=1e-4)
    assert P[0, 3] == pytest.approx(0.0, abs=1e-4)

    # Check energy delivery bounds
    energy_step_0 = P[0, 0] * dt_hours
    assert energy_step_0 <= 0.6 + 1e-4, "Schedule charged before arrival"
    energy_step_1 = P[0, 1] * dt_hours
    assert energy_step_1 <= 0.6 + 1e-4, "Schedule counted energy after departure"

    total_energy = float(np.sum(P[0, :]) * dt_hours)
    assert total_energy == pytest.approx(1.2, abs=1e-3)
    assert res.unmet_energy_kwh[0] == pytest.approx(3.8, abs=1e-3)

    # Validate schedule
    base_load_arr = np.array(feeder.base_load_profile_kw[:num_steps])
    valid, msg = validate_schedule_constraints(
        power_matrix=P,
        unmet_arr=res.unmet_energy_kwh,
        sessions=[session],
        base_load_arr=base_load_arr,
        transformer_limit_kw=feeder.transformer_limit_kw,
        dt_hours=dt_hours,
        num_steps=num_steps,
    )
    assert valid, f"Schedule validation failed: {msg}"

    # Simulator physical integration
    sim_session = EVSession(10, 20, energy_req_kwh=5.0, max_rate_kw=7.2, session_id="EV_SIM")
    sim = ChargingStationSimulator(feeder=feeder, sessions=[sim_session], time_step_min=time_step_min)
    sim.set_ev_setpoint_kw("EV_SIM", 7.2)
    assert sim.get_session_power_kw("EV_SIM") == pytest.approx(2.4, abs=1e-4)
    sim.step()
    assert sim_session.delivered_energy_kwh == pytest.approx(0.6, abs=1e-4)
    sim.set_ev_setpoint_kw("EV_SIM", 7.2)
    assert sim.get_session_power_kw("EV_SIM") == pytest.approx(2.4, abs=1e-4)
    sim.step()
    assert sim_session.delivered_energy_kwh == pytest.approx(1.2, abs=1e-4)
    sim.set_ev_setpoint_kw("EV_SIM", 7.2)
    assert sim.get_session_power_kw("EV_SIM") == pytest.approx(0.0, abs=1e-4)

    # Uncontrolled charging baseline integration
    s_unc = EVSession(10, 20, energy_req_kwh=5.0, max_rate_kw=7.2, session_id="EV_UNC")
    res_unc = run_uncontrolled_charging(feeder=feeder, sessions=[s_unc], time_step_min=time_step_min, num_steps=num_steps)
    assert res_unc.sessions[0].delivered_energy_kwh == pytest.approx(1.2, abs=1e-4)
