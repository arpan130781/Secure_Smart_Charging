"""Unit tests for attacks.py (Task 10)."""

import pytest
from ev_sched.hal import MeterInterface
from ev_sched.attacks import (
    ConstantBiasMeter,
    RampAttackMeter,
    ReplayAttackMeter,
    create_attacked_meter,
)


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
