"""Unit tests for attacks.py (Task 10)."""

import pytest

from ev_sched.attacks import (
    ConstantBiasMeter,
    RampAttackMeter,
    ReplayAttackMeter,
    create_attacked_meter,
)
from ev_sched.hal import EVSEInterface, MeterInterface


class MockMeter(MeterInterface):
    """Simple mock meter for deterministic testing."""

    def __init__(self, values: dict[int, float] | None = None) -> None:
        self.values = values or {}

    def read_feeder_kw(self, t_min: int) -> float:
        return self.values.get(t_min, 80.0)


def test_constant_bias_meter() -> None:
    inner = MockMeter()
    attacked = ConstantBiasMeter(inner_meter=inner, start_min=100, duration_min=60, delta_kw=25.0)

    # Before attack: true reading (80.0 kW)
    assert attacked.read_feeder_kw(90) == 80.0

    # Inside attack: 80.0 - 25.0 = 55.0 kW
    assert attacked.read_feeder_kw(100) == 55.0
    assert attacked.read_feeder_kw(159) == 55.0

    # After attack: 80.0 kW
    assert attacked.read_feeder_kw(160) == 80.0


def test_ramp_attack_meter() -> None:
    inner = MockMeter()
    # 60 min duration, ramp to 30 kW -> 0.5 kW / min
    attacked = RampAttackMeter(inner_meter=inner, start_min=100, duration_min=60, max_delta_kw=30.0)

    # Before
    assert attacked.read_feeder_kw(90) == 80.0

    # Start: delta = 0 -> 80.0 kW
    assert attacked.read_feeder_kw(100) == 80.0

    # Halfway (t = 130, elapsed 30 min): delta = 15 kW -> 65.0 kW
    assert attacked.read_feeder_kw(130) == 65.0

    # End of window (t = 159): delta ~ 30 kW
    assert pytest.approx(attacked.read_feeder_kw(159), 0.5) == 50.0

    # After
    assert attacked.read_feeder_kw(160) == 80.0


def test_replay_attack_meter() -> None:
    values = {0: 20.0, 15: 25.0, 30: 30.0, 45: 35.0, 100: 90.0, 115: 95.0}
    inner = MockMeter(values)
    attacked = ReplayAttackMeter(
        inner_meter=inner,
        start_min=100,
        duration_min=30,
        replay_source_start_min=0,
    )

    # Prime buffer
    attacked.read_feeder_kw(0)
    attacked.read_feeder_kw(15)

    # Replay window: at t=100, replays t=0 (20.0 kW instead of 90.0 kW)
    assert attacked.read_feeder_kw(100) == 20.0
    # At t=115, replays t=15 (25.0 kW instead of 95.0 kW)
    assert attacked.read_feeder_kw(115) == 25.0


def test_create_attacked_meter_factory() -> None:
    inner = MockMeter()
    m_bias = create_attacked_meter(inner, {"attack_type": "constant_bias", "start_min": 10, "duration_min": 20, "magnitude_kw": 15.0})
    assert isinstance(m_bias, ConstantBiasMeter)

    m_ramp = create_attacked_meter(inner, {"attack_type": "ramp", "start_min": 10, "duration_min": 20, "magnitude_kw": 15.0})
    assert isinstance(m_ramp, RampAttackMeter)

    m_replay = create_attacked_meter(inner, {"attack_type": "replay", "start_min": 10, "duration_min": 20, "replay_source_start_min": 0})
    assert isinstance(m_replay, ReplayAttackMeter)

    m_adapt = create_attacked_meter(
        inner,
        {
            "attack_type": "adaptive",
            "start_min": 10,
            "duration_min": 20,
            "base_forecast_profile_kw": [40.0] * 10,
        },
    )
    assert hasattr(m_adapt, "is_attack_active")


def test_adaptive_attack_evades_cusum_detector() -> None:
    """Task 17: Verify adaptive attacker evades Detector 1 despite physical overload."""
    from ev_sched.attacks import AdaptiveAttackMeter
    from ev_sched.detector import CUSUMDetector

    # Physical reality: an unmodeled industrial load surge causes true base load
    # to spike to 85 kW. True physical feeder load = 85 kW + 30 kW EV = 115 kW.
    # Feeder limit is 100 kW -> True overload = 15 kW!
    inner_meter = MockMeter({100: 115.0, 115: 115.0, 130: 115.0})

    # The EMS model holds a base forecast of 50.0 kW.
    # With 30.0 kW EV load, expected feeder load = 80.0 kW.
    base_forecast = [50.0] * 20

    class MockSimulator:
        def get_total_ev_power_kw(self) -> float:
            return 30.0

    sim = MockSimulator()
    adaptive_meter = AdaptiveAttackMeter(
        inner_meter=inner_meter,
        start_min=100,
        duration_min=45,
        base_forecast_profile_kw=base_forecast,
        simulator=sim,
        noise_std_kw=0.1,
        seed=42,
    )

    detector = CUSUMDetector(threshold=18.0, slack_kw=1.5, instant_threshold_kw=20.0)

    # During attack (t = 100, 115, 130 min)
    for t_min in [100, 115, 130]:
        reported_kw = adaptive_meter.read_feeder_kw(t_min)
        expected_kw = 50.0 + 30.0  # 80.0 kW
        alert = detector.update(reported_feeder_kw=reported_kw, expected_feeder_kw=expected_kw)
        # Residual stays within small noise (|r| < 1.0 kW << 18.0 kW threshold)
        assert abs(alert.residual_kw) < 1.0
        # CUSUM is completely blind: NO ALARM raised!
        assert not alert.is_alarm

        # Yet physical meter reading is 115.0 kW (severe overload!)
        assert inner_meter.read_feeder_kw(t_min) == 115.0

    # Test HAL EVSE interface integration
    class MockEVSE(EVSEInterface):
        def set_setpoint_kw(self, ev_id: str, power_kw: float) -> None:
            pass

        def read_session_kw(self, ev_id: str) -> float:
            return 15.0

    evse = MockEVSE()
    adaptive_meter_evse = AdaptiveAttackMeter(
        inner_meter=inner_meter,
        start_min=100,
        duration_min=45,
        base_forecast_profile_kw=base_forecast,
        evse=evse,
        session_ids=["EV_1", "EV_2"],
        noise_std_kw=0.0,
        seed=42,
    )
    # 50 kW base + 2 * 15 kW EV = 80 kW
    assert adaptive_meter_evse.read_feeder_kw(100) == 80.0

