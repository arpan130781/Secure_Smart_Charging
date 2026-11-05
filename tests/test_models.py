"""Unit tests for models.py (EVSession, Feeder, Tariff)."""

import pytest

from ev_sched.models import EVSession, Feeder, Tariff


def test_ev_session_valid() -> None:
    session = EVSession(
        arrival_min=480,       # 08:00 AM (480 min)
        departure_min=1020,    # 17:00 PM (1020 min)
        energy_req_kwh=32.5,   # 32.5 kWh
        max_rate_kw=11.0,      # 11.0 kW
        session_id="EV_01",
    )
    assert session.arrival_min == 480
    assert session.departure_min == 1020
    assert session.energy_req_kwh == 32.5
    assert session.max_rate_kw == 11.0
    assert session.duration_min == 540  # 9 hours = 540 minutes
    assert session.remaining_energy_req_kwh == 32.5
    assert session.is_connected(500)
    assert not session.is_connected(400)
    assert not session.is_connected(1020)


def test_ev_session_validation_errors() -> None:
    with pytest.raises(ValueError, match="strictly greater"):
        EVSession(arrival_min=500, departure_min=500, energy_req_kwh=20.0, max_rate_kw=7.2)

    with pytest.raises(ValueError, match="strictly greater"):
        EVSession(arrival_min=600, departure_min=400, energy_req_kwh=20.0, max_rate_kw=7.2)

    with pytest.raises(ValueError, match="cannot be negative"):
        EVSession(arrival_min=100, departure_min=200, energy_req_kwh=-5.0, max_rate_kw=7.2)

    with pytest.raises(ValueError, match="must be positive"):
        EVSession(arrival_min=100, departure_min=200, energy_req_kwh=20.0, max_rate_kw=0.0)


def test_feeder_model() -> None:
    feeder = Feeder(transformer_limit_kw=150.0, base_load_profile_kw=[50.0] * 96)
    assert feeder.transformer_limit_kw == 150.0
    assert len(feeder.base_load_profile_kw) == 96

    with pytest.raises(ValueError, match="must be positive"):
        Feeder(transformer_limit_kw=0.0)


def test_tariff_model() -> None:
    tariff = Tariff(price_per_kwh_per_15min=[0.15] * 96)
    assert len(tariff.price_per_kwh_per_15min) == 96
    assert tariff.price_per_kwh_per_15min[0] == 0.15
